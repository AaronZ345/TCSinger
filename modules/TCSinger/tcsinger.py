import torch.nn as nn
from modules.tts.fs import FastSpeech
import math
from modules.tts.commons.align_ops import expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
import torch
from modules.TCSinger.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion, GaussianMultinomialDiffusionx0
from modules.TCSinger.diff.net import DDiffNet
from utils.commons.hparams import hparams
from modules.TCSinger.style_encoder import StyleEncoder
from modules.commons.layers import Embedding
from modules.TCSinger.style_adapt_fft import SAFFT
from modules.TCSinger.style_adapt_decoder import SADecoder
from modules.TCSinger.style_adapt_postnet import StylePostnet
import torch.nn.functional as F


# denoise
DIFF_DECODERS = {
    'sad': lambda hp: SAFFT(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}

# process note_tokens, note_durs and note types
class NoteEncoder(nn.Module):
    def __init__(self, n_vocab, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)
        self.type_emb = nn.Embedding(5, hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.type_emb.weight, 0.0, hidden_channels ** -0.5)
        self.dur_ln = nn.Linear(1, hidden_channels)

    def forward(self, note_tokens, note_durs, note_types):
        x = self.emb(note_tokens) * math.sqrt(self.hidden_channels)
        types = self.type_emb(note_types) * math.sqrt(self.hidden_channels)
        durs = self.dur_ln(note_durs.unsqueeze(dim=-1))
        x = x + durs + types
        return x


# main model
class TCSinger(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        
        # note encoder
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)
        
        # predict dur in sdlm
        del self.dur_predictor
        
        # style encoder
        self.style_encoder = StyleEncoder(hparams)
        
        # f0 predictor
        if hparams["f0_gen"] == "gmdiff":
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            if hparams["param_"] == "x0":
                self.f0_gen = GaussianMultinomialDiffusionx0(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"])
            else:
                self.f0_gen = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"])

        # decoder
        if hparams['de']=='sad':
            self.decoder=SADecoder(
                out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
                timesteps=hparams['timesteps'],
                time_scale=hparams['timescale'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
            )

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None,target=None,ph_lengths=None,
                f0=None, uv=None, infer=False, note=None, note_dur=None, note_type=None):
        ret = {}
        
        # content
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        encoder_out = encoder_out + note_out
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        
        # spk embed
        if spk_embed!=None:
            ret['spk_embed']=spk_embed = self.forward_style_embed(spk_embed, spk_id)
        else:
            ret['spk_embed']=spk_embed = self.style_encoder.encode_spk_embed(target.transpose(1, 2)).transpose(1, 2)
        
        # style embed
        in_nonpadding = (mel2ph > 0).float()[:, :, None]  
        style, vq_loss,indices = self.style_encoder.encode_style(target.transpose(1, 2), target.transpose(1, 2), in_nonpadding.transpose(1,2), in_nonpadding.transpose(1,2), mel2ph, src_nonpadding.transpose(1,2), ph_lengths)    
        ret['vq_loss']=vq_loss
        ret['style']=style =expand_states(style.transpose(1, 2), mel2ph)

        # length regulator
        ret['mel2ph']=mel2ph
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)
        ret['content'] = decoder_inp

        # add pitch embed
        midi_notes = None
        pitch_inp = (decoder_inp + spk_embed + style) * tgt_nonpadding
        if infer:
            f0, uv = None, None
            midi_notes = expand_states(note[:, :, None], mel2ph)
        decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, midi_notes=midi_notes)

        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed + style) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(txt_tokens,target,tgt_nonpadding, ret, infer=infer,decoder_inp=decoder_inp)
        return ret
    
    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, **kwargs):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
            
        if hparams["f0_gen"] == "gmdiff":
            f0, uv = self.add_gmdiff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)

        f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        
        f0_denorm_smooth = self.smooth_f0(f0_denorm, window_size=5)

        # Convert smoothed f0 to pitch
        pitch = f0_to_coarse(f0_denorm_smooth)  # start from 0 [B, T_txt]
        ret['f0_denorm_pred'] = f0_denorm_smooth
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def add_gmdiff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, **kwargs):
        if f0 is None:
            infer = True
        else:
            infer = False
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            lower_bound = midi_notes - 3 # 1 for good gtdur F0RMSE
            upper_bound = midi_notes + 3 # 1 for good gtdur F0RMSE
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            
            pitch_pred = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)
        return f0, uv

    def forward_decoder(self,txt_tokens,ref_mels, tgt_nonpadding, ret, infer,decoder_inp, **kwargs):
        if hparams['de']!='sad':
            x = decoder_inp  # [B, T, H]
            x = self.decoder(x)
            x = self.mel_out(x)
            return x * tgt_nonpadding
        else:
            ret=self.decoder(txt_tokens, ret, ref_mels, infer)
            x=ret['mel_out']
            return x * tgt_nonpadding
        
    def smooth_f0(self, f0, window_size=5):
        # Ensure window_size is odd for symmetric padding
        padding = window_size // 2
        # Use 1D average pooling for smoothing
        f0_padded = F.pad(f0, (padding, padding), mode='reflect')
        f0_smooth = F.avg_pool1d(f0_padded.unsqueeze(1), kernel_size=window_size, stride=1).squeeze(1)
        return f0_smooth


# postnet
class SAPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        cond_hs=80+hparams['hidden_size']*3
        if hparams['use_spk_id']:
            self.spk_id_proj = Embedding(hparams['num_spk'], hparams['hidden_size'])
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, hparams['hidden_size'], bias=True)
        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        self.sapost = StylePostnet(
            phone_encoder=None,
            out_dims=80, denoise_fn=DIFF_DECODERS[hparams['post_diff_decoder_type']](hparams),
            timesteps=hparams['post_timesteps'],
            K_step=hparams['post_K_step'],
            loss_type=hparams['post_diff_loss_type'],
            spec_min=hparams['post_spec_min'], spec_max=hparams['post_spec_max'],
        )
    
    def forward(self, tgt_mels, infer, ret, spk_embed=None):
        x_recon = ret['mel_out']
        g = x_recon.detach()
        B, T, _ = g.shape
        content=ret['content']
        g = torch.cat([g, content], dim=-1)
        if spk_embed!=None:       
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        else:
            spk_embed=ret['spk_embed']
        spk_embed = spk_embed.repeat(1, T, 1)
        style=ret['style']
        g = torch.cat([g, spk_embed,style], dim=-1)
        g = self.ln_proj(g)
        self.sapost(g, tgt_mels, x_recon, ret, infer)
