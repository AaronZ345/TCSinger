import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from modules.commons.transformer import SinusoidalPositionalEmbedding
import torch
from torch import nn
import torch.nn.functional as F
from modules.commons.transformer import DEFAULT_MAX_TARGET_POSITIONS,TransformerEncoderLayer,MultiheadAttention,TransformerFFNLayer
from utils.commons.hparams import hparams
import math

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SpatialNorm(nn.Module):
    def __init__(self, f_channels, zq_channels, freeze_norm_layer=False, add_conv=True, **norm_layer_params):
        super().__init__()
        self.norm_layer = nn.LayerNorm(f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv1d(zq_channels, zq_channels,1)
        self.conv_y = nn.Conv1d(zq_channels, f_channels, 1)
        self.conv_b = nn.Conv1d(zq_channels, f_channels, 1)
    def forward(self, f, zq):
        norm_f = self.norm_layer(f).transpose(1, 2)
        f=f.transpose(1, 2)
        zq=zq.transpose(0, 1).transpose(1, 2)
        f_size = f.shape[-1:]
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        new_f=new_f.transpose(1, 2)
        return new_f


class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            # self.layer_norm1 = nn.LayerNorm(c)
            self.layer_norm1=SpatialNorm(c,c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        # self.layer_norm2 = nn.LayerNorm(c)
        self.layer_norm2=SpatialNorm(c,c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self, x,zq, encoder_padding_mask=None, **kwargs):
        # layer_norm_training = kwargs.get('layer_norm_training', None)
        layer_norm_training=self.training
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x,zq)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x,zq)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size)

    def forward(self, x,zq, **kwargs):
        return self.op(x,zq, **kwargs)


# denoise_fn
class VQFFT(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=0.0,
                 num_heads=2, use_pos_embed=True, use_last_norm=True,
                 use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            self.layer_norm=SpatialNorm(self.hidden_size,self.hidden_size)
            # self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, cond=None,padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x,cond, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x,cond) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x
    
class VQFFT2(VQFFT):
    def __init__(self, hidden_size=256, num_layers=4, kernel_size=9, num_heads=2):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)
        

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SAFFT(VQFFT2):
    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)
        dim = hparams['residual_channels']
        self.input_projection = Conv1d(hparams['audio_num_mel_bins'], dim, 1)
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.get_mel_out = nn.Linear(hparams['hidden_size'], 80, bias=True)
        self.get_decode_inp = nn.Linear(hparams['hidden_size'] + dim + dim,
                                     hparams['hidden_size'])  # hs + dim + 80 -> hs

    def forward(self, spec, diffusion_step, cond,style, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param spec: [B, 1, 80, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x).permute([0, 2, 1])  #  [B, T, residual_channel]
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)  # [B, dim]
        cond = cond.permute([0, 2, 1])  # [B, T, M]

        seq_len = cond.shape[1]  # [T_mel]
        time_embed = diffusion_step[:, None, :]  # [B, 1, dim]
        time_embed = time_embed.repeat([1, seq_len, 1])  # # [B, T, dim]

        decoder_inp = torch.cat([x, cond, time_embed], dim=-1)  # [B, T, dim + H + dim]
        decoder_inp = self.get_decode_inp(decoder_inp)  # [B, T, H]
        x = decoder_inp

        '''
        Required x: [B, T, C]
        :return: [B, T, C] or [L, B, T, C]
        '''
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x,style, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x,style) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]

        x = self.get_mel_out(x).permute([0, 2, 1])  # [B, 80, T]
        return x[:, None, :, :]