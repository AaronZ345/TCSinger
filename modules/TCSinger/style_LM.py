import torch
from torch import nn
import torch.nn.functional as F
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.layers import LayerNorm, Linear
from modules.commons.transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from utils.commons.hparams import hparams
from modules.TCSinger.tcsinger import NoteEncoder
import math


# process technique and global style
class TechEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.mix_emb = nn.Embedding(3, hidden_channels, padding_idx=0)
        self.falsetto_emb = nn.Embedding(3, hidden_channels, padding_idx=0)
        self.breathy_emb = nn.Embedding(3, hidden_channels, padding_idx=0)
        self.pharyngeal_emb = nn.Embedding(3, hidden_channels, padding_idx=0)
        self.glissando_emb = nn.Embedding(3, hidden_channels, padding_idx=0)
        self.vibrato_emb = nn.Embedding(3, hidden_channels, padding_idx=0)

        # 保存字符串到索引的映射
        self.emotion2idx = {'happy': 1, 'sad': 2, 'no': 3}
        self.singing_method2idx = {'bel canto': 1, 'pop': 2, 'no': 3}
        self.pace2idx = {'moderate': 1, 'fast': 2, 'slow': 3, 'no': 4}
        self.range2idx = {'low': 1, 'medium': 2, 'high': 3, 'no': 4}

        self.emotion_emb = nn.Embedding(len(self.emotion2idx) + 1, hidden_channels, padding_idx=0)
        self.singing_method_emb = nn.Embedding(len(self.singing_method2idx) + 1, hidden_channels, padding_idx=0)
        self.pace_emb = nn.Embedding(len(self.pace2idx) + 1, hidden_channels, padding_idx=0)
        self.range_emb = nn.Embedding(len(self.range2idx) + 1, hidden_channels, padding_idx=0)

        # 初始化嵌入层权重
        for emb in [self.mix_emb, self.falsetto_emb, self.breathy_emb, self.pharyngeal_emb,
                    self.glissando_emb, self.vibrato_emb, self.emotion_emb,
                    self.singing_method_emb, self.pace_emb, self.range_emb]:
            nn.init.normal_(emb.weight, 0.0, hidden_channels ** -0.5)

    def forward(self, mix,falsetto,breathy,pharyngeal,glissando,vibrato,emotion,singing_method,pace,range_):
        batch_size = mix.size(0)
        device = mix.device

        mix = self.mix_emb(mix) * math.sqrt(self.hidden_channels)
        falsetto = self.falsetto_emb(falsetto) * math.sqrt(self.hidden_channels)
        breathy = self.breathy_emb(breathy) * math.sqrt(self.hidden_channels)
        pharyngeal = self.pharyngeal_emb(pharyngeal) * math.sqrt(self.hidden_channels)
        glissando = self.glissando_emb(glissando) * math.sqrt(self.hidden_channels)
        vibrato = self.vibrato_emb(vibrato) * math.sqrt(self.hidden_channels)

        # Map each string in the batch to its corresponding index
        emotion_idx = [self.emotion2idx.get(e, 0) for e in emotion]
        singing_method_idx = [self.singing_method2idx.get(sm, 0) for sm in singing_method]
        pace_idx = [self.pace2idx.get(p, 0) for p in pace]
        range_idx = [self.range2idx.get(r, 0) for r in range_]

        # Convert lists of indices to tensors
        emotion_idx = torch.tensor(emotion_idx, device=device)
        singing_method_idx = torch.tensor(singing_method_idx, device=device)
        pace_idx = torch.tensor(pace_idx, device=device)
        range_idx = torch.tensor(range_idx, device=device)

        emotion_emb = self.emotion_emb(emotion_idx).unsqueeze(1) * math.sqrt(self.hidden_channels)
        singing_method_emb = self.singing_method_emb(singing_method_idx).unsqueeze(1) * math.sqrt(self.hidden_channels)
        pace_emb = self.pace_emb(pace_idx).unsqueeze(1) * math.sqrt(self.hidden_channels)
        range_emb = self.range_emb(range_idx).unsqueeze(1) * math.sqrt(self.hidden_channels)

        x = mix + falsetto + breathy + pharyngeal + glissando + vibrato + emotion_emb + singing_method_emb + pace_emb + range_emb

        return x

# SDLM
class StyleLanguageModel(nn.Module):
    def __init__(self, dict_size):
        super().__init__()
        self.hidden_size = hidden_size = hparams['lm_hidden_size']
        self.ph_encoder = RelTransformerEncoder(
        dict_size, hidden_size, hidden_size,
        hidden_size*4, hparams['num_heads'], hparams['enc_layers'],
        hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.vqcode_emb = Embedding(hparams['vq_ph_codebook_dim'] + 2, hidden_size, 0)

        self.tech_encoder = TechEncoder(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        self.embed_positions2 = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        dec_num_layers = 8

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(hidden_size, 0., kernel_size=5, num_heads=8) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)
        self.project_out_dim = Linear(hidden_size, hparams['vq_ph_codebook_dim'] + 1, bias=True)

        self.layers2 = nn.ModuleList([])
        self.layers2.extend([
            TransformerDecoderLayer(hidden_size, 0., kernel_size=5, num_heads=8) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm2 = LayerNorm(hidden_size)

        n_chans = self.hidden_size
        self.linear_in = Linear(1, n_chans, bias=True)
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

        # Speaker embed related
        self.spk_embed_proj = Linear(hparams['hidden_size'], hidden_size, bias=True)
        self.spk_mode = 'direct' # 'direct' or 'attn'
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)

    def forward(self, ph_tokens, prev_vq_code, prev_dur, spk_embed,note=None, note_dur=None, note_type=None, incremental_state=None, ret=None, 
                mix=None,falsetto=None,breathy=None,pharyngeal=None,glissando=None,vibrato=None,
                emotion=None,pace=None,range_=None,singing_method=None, control = False, *args,**kwargs):
        # run encoder
        if control:
            batch_size, seq_len = ph_tokens.size(0), ph_tokens.size(1)
            x = torch.zeros(batch_size, seq_len, self.hidden_size).to(ph_tokens.device)
            y = torch.zeros(batch_size, seq_len, self.hidden_size).to(ph_tokens.device)
            tech_embed = self.tech_encoder(mix,falsetto,breathy,pharyngeal,glissando,vibrato,emotion,singing_method,pace,range_)
        else:
            x = self.vqcode_emb(prev_vq_code)
            y=prev_dur.unsqueeze(-1)
            y=self.linear_in(y)
            
        src_nonpadding = (ph_tokens > 0).float()[:, :, None]

        encoder_out = self.ph_encoder(ph_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        ph_embed = (encoder_out + note_out) * src_nonpadding
        
        if self.spk_mode == 'direct':
            ph_embed = ph_embed + self.spk_embed_proj(spk_embed)
            if control:
                ph_embed = ph_embed + tech_embed
            ph_embedx=ph_embed + y
            ph_embedy=ph_embed + x
            ph_embed = ph_embed * src_nonpadding
            ph_embedx = ph_embedx * src_nonpadding
            ph_embedy = ph_embedy * src_nonpadding

        # run style decoder
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            ph_embed = ph_embedx[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            positions = positions[:, -1:]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            self_attn_padding_mask = ph_tokens.eq(0).data

        x += positions
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        ph_embed = ph_embedx.transpose(0, 1)
        x = x + ph_embed

        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)

        # run dur decoder
        if incremental_state is not None:
            positions = self.embed_positions2(
                prev_dur,
                incremental_state=incremental_state
            )
            ph_embed = ph_embedy[:, y.shape[1] - 1:y.shape[1]]
            y = y[:, -1:]
            positions = positions[:, -1:]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions2(
                prev_dur,
                incremental_state=incremental_state
            )
            self_attn_padding_mask = ph_tokens.eq(0).data

        y += positions
        # B x T x C -> T x B x C
        y = y.transpose(0, 1)
        ph_embed = ph_embedy.transpose(0, 1)
        y = y + ph_embed

        for layer in self.layers2:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(y)
            else:
                self_attn_mask = None

            y, attn_logits = layer(
                y,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        y = self.layer_norm2(y)
        # T x B x C -> B x T x C
        y = y.transpose(0, 1)
        y = self.linear(y)  
        y = y * src_nonpadding  # (B, T, C)

        return x, y

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(self.fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def infer(self, ph_tokens, ph_vqcode, indur, spk_embed, prompt_length, ret, note=None, note_dur=None, note_type=None, mode='argmax', *args,**kwargs):
        incremental_state = None
        # Add prompt
        vq_decoded = torch.zeros_like(ph_tokens)
        dur_decoded = torch.zeros_like(ph_tokens).float()
        
        vq_decoded[:, :prompt_length] = ph_vqcode[:, :prompt_length]
        dur_decoded[:, :prompt_length] = indur[:, :prompt_length]

        # Start Decode
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['vq_ph_codebook_dim'] + 1)
        dur_decoded = F.pad(dur_decoded, [1, 0], value=0)
        if mode == 'argmax':
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                vq_pred, dur_pred = self(ph_tokens, vq_decoded[:, :-1],dur_decoded[:, :-1], spk_embed,note=note, note_dur=note_dur, note_type=note_type,
                            incremental_state=incremental_state, ret=ret, control=False)
                vq_pred = torch.argmax(F.softmax(vq_pred, dim=-1), -1)
                dur_pred = dur_pred[..., 0]  # (B, Tmax)
                vq_decoded[:, step + 1] = vq_pred[:, step]
                dur_decoded[:, step + 1] = dur_pred[:, step]
        elif mode == 'topk':
            K = 10
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                vq_pred, dur_pred = self(ph_tokens, vq_decoded[:, :-1],dur_decoded[:, :-1], spk_embed,note=note, note_dur=note_dur, note_type=note_type,
                            incremental_state=incremental_state, ret=ret, control=False)
                _, idx = F.softmax(vq_pred, dim=-1).topk(k = K, axis = -1)
                rand_idx = random.randint(0,K-1)
                vq_decoded[:, step + 1] = idx[:, step, rand_idx]

                _, idx = F.softmax(dur_pred, dim=-1).topk(k = K, axis = -1)
                rand_idx = random.randint(0,K-1)
                dur_decoded[:, step + 1] = idx[:, step, rand_idx]
        return vq_decoded[:, 1:],dur_decoded[:,1:]
    
    def infer_control(self, ph_tokens, spk_embed, ret, note=None, note_dur=None, note_type=None, mode='argmax',
                      mix=None,falsetto=None,breathy=None,pharyngeal=None,glissando=None,vibrato=None,emotion=None,pace=None,range_=None,singing_method=None, *args,**kwargs):
        incremental_state = None
        # Add prompt
        vq_decoded = torch.zeros_like(ph_tokens)
        dur_decoded = torch.zeros_like(ph_tokens).float()
        batch_size, seq_len = ph_tokens.size(0), ph_tokens.size(1)
        
        vq_pred, dur_pred = self(ph_tokens, vq_decoded,dur_decoded, spk_embed,note=note, note_dur=note_dur, note_type=note_type,
                        incremental_state=incremental_state, ret=ret, mix=mix, falsetto=falsetto, breathy=breathy,
                        pharyngeal=pharyngeal, glissando=glissando, vibrato=vibrato,
                        emotion=emotion, pace=pace, range_=range_, singing_method=singing_method,
                        control=True)
        
        if mode == 'argmax':
            # 对 vq_pred 应用 softmax，然后取 argmax
            vq_pred_probs = F.softmax(vq_pred, dim=-1)
            vq_decoded = torch.argmax(vq_pred_probs, dim=-1)  # 形状为 (B, T)

        elif mode == 'topk':
            K = 10
            vq_pred_probs = F.softmax(vq_pred, dim=-1)
            topk_probs, topk_indices = vq_pred_probs.topk(k=K, dim=-1)

            random_indices = torch.randint(0, K, (batch_size, seq_len), device=ph_tokens.device)
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(ph_tokens.device)
            time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(ph_tokens.device)
            vq_decoded = topk_indices[batch_indices, time_indices, random_indices]

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        dur_decoded = dur_pred[..., 0]  
        return vq_decoded, dur_decoded

    def fill_with_neg_inf2(self, t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(-1e8).type_as(t)