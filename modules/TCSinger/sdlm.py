import random
import torch.nn.functional as F
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.commons.hparams import hparams
from modules.TCSinger.tcsinger import TCSinger
from modules.TCSinger.style_LM import StyleLanguageModel


# final model with sdlm
class SDLM(TCSinger):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        self.sd_lm = StyleLanguageModel(ph_dict_size)

    def forward(self, txt_tokens, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,ref_dur=None,tgt_dur=None,
                mel_prompt=None, spk_embed_prompt=None, note=None, note_dur=None, note_type=None,f0=None,uv=None,spk_id=None,
                mix=None,falsetto=None,breathy=None,pharyngeal=None,glissando=None,vibrato=None,
                emotion=None,pace=None,range_=None,singing_method=None,*args,**kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # Forward SDLM
        if not infer:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # Get GT VQCode
            ph_vqcode = self.style_encoder.encode_ph_vqcode(tgt_mels.transpose(1,2), in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            if spk_embed_prompt!=None:
                ret['spk_embed']= spk_embed = self.forward_style_embed(spk_embed_prompt, spk_id)           
            else:
                ret['spk_embed']= spk_embed = self.style_encoder.encode_spk_embed(mel_prompt.transpose(1, 2)).transpose(1, 2)
            # Forward SDLM
            ph_vqcode = (ph_vqcode.detach() + 1) * src_nonpadding.squeeze(-1).long()
            prev_ph_vqcode = F.pad(ph_vqcode[:, :-1], [1, 0], value=hparams['vq_ph_codebook_dim'] + 1)

            tgt_dur = tgt_dur.detach() * src_nonpadding.squeeze(-1).float()
            prev_dur = F.pad(tgt_dur[:, :-1], [1, 0], value=0)

            # control or transfer
            if 'happy' in emotion or 'sad' in emotion:
                control = True if random.random() < 0.5 else False
            else:
                control = False
            vq_codes_pred,dur_pred = self.sd_lm(txt_tokens, prev_ph_vqcode,prev_dur, spk_embed, 
                                    note=note,note_dur=note_dur,note_type=note_type, ret=ret, 
                                    mix=mix,falsetto=falsetto,breathy=breathy,pharyngeal=pharyngeal,glissando=glissando,vibrato=vibrato,
                                    emotion=emotion,pace=pace,range_=range_,singing_method=singing_method,control=control)

            src_padding = txt_tokens == 0
            dur_pred = dur_pred[..., 0]  # (B, Tmax)
            dur_pred=dur_pred * (1 - src_padding.float())
            ret['vq_codes_pred'], ret['vq_codes'] = vq_codes_pred, ph_vqcode
            ret['dur'] = dur_pred

        else: # infer
            # content
            encoder_out = self.encoder(txt_tokens)  # [B, T, C]
            note_out = self.note_encoder(note, note_dur, note_type)
            encoder_out = encoder_out + note_out
            if spk_embed_prompt != None:
                spk_embed = self.forward_style_embed(spk_embed_prompt, spk_id) 
            else:
                ret['spk_embed'] = spk_embed = self.style_encoder.encode_spk_embed(mel_prompt.transpose(1, 2)).transpose(1, 2)

            # infer vq code
            in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
            ph_vqcode = self.style_encoder.encode_ph_vqcode(mel_prompt.transpose(1,2), in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
            ph_vqcode = (ph_vqcode.detach() + 1)* prompt_src_nonpadding.squeeze(-1).long()
            indur = ref_dur.detach()* prompt_src_nonpadding.squeeze(-1).float()
            vq_codes_pred,dur_pred = self.sd_lm.infer(txt_tokens, ph_vqcode, indur, spk_embed, txt_tokens_prompt.shape[1],
                                    note=note,note_dur=note_dur,note_type=note_type,ret=ret)
            style = self.style_encoder.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
            
            # add dur
            dur = dur_pred
            src_padding = txt_tokens == 0
            dur = dur * (1 - src_padding.float())
            if mel2ph is None:
                dur = (dur.exp() - 1).clamp(min=0)
                mel2ph = self.length_regulator(dur, src_padding).detach()
            mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
            style = expand_states(style.transpose(1, 2), mel2ph)

            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
            decoder_inp = expand_states(encoder_out, mel2ph)

            # add pitch embed
            midi_notes = None
            pitch_inp = (decoder_inp + spk_embed + style) * tgt_nonpadding
            f0, uv = None, None
            midi_notes = expand_states(note[:, :, None], mel2ph)
            
            ret['content'] = decoder_inp = decoder_inp[:, mel2ph_prompt.shape[1]:, :]
            ret['style'] = style = style[:,mel2ph_prompt.shape[1]:,:]
            ret['mel2ph'] = mel2ph_tgt = mel2ph[:,mel2ph_prompt.shape[1]:]
            midi_notes = midi_notes[:,mel2ph_prompt.shape[1]:,:]
            pitch_inp = pitch_inp[:,mel2ph_prompt.shape[1]:,:]
            tgt_nonpadding = tgt_nonpadding[:,mel2ph_prompt.shape[1]:,:]

            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph_tgt, ret, midi_notes=midi_notes)

            # decoder input
            ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed + style) * tgt_nonpadding
            ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer)

        return ret

    def infer(self, txt_tokens, txt_tokens_prompt, mel2ph_prompt=None, ref_dur=None,
                mel_prompt=None, spk_embed_prompt=None, note=None, note_dur=None, note_type=None,f0=None,uv=None,spk_id=None,
                mix=None,falsetto=None,breathy=None,pharyngeal=None,glissando=None,vibrato=None,
                emotion=None,pace=None,range_=None,singing_method=None,control=False,*args,**kwargs):
            ret = {}
            
            # content
            encoder_out = self.encoder(txt_tokens)  # [B, T, C]
            note_out = self.note_encoder(note, note_dur, note_type)
            encoder_out = encoder_out + note_out
            
            # spk embed
            if spk_embed_prompt != None:
                spk_embed = self.forward_style_embed(spk_embed_prompt, spk_id)
            else:
                ret['spk_embed'] = spk_embed = self.style_encoder.encode_spk_embed(mel_prompt.transpose(1, 2)).transpose(1, 2)
            
            if control:
                vq_codes_pred,dur_pred = self.sd_lm.infer_control(txt_tokens, spk_embed, ret,
                                    note=note,note_dur=note_dur,note_type=note_type,
                                    mix=mix,falsetto=falsetto,breathy=breathy,pharyngeal=pharyngeal,glissando=glissando,vibrato=vibrato,
                                    emotion=emotion,pace=pace,range_=range_,singing_method=singing_method)
                style = self.style_encoder.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
            else:
                prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]
                in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
                ph_vqcode = self.style_encoder.encode_ph_vqcode(mel_prompt.transpose(1,2), in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
                ph_vqcode = (ph_vqcode.detach() + 1)* prompt_src_nonpadding.squeeze(-1).long()
                indur = ref_dur.detach()* prompt_src_nonpadding.squeeze(-1).float()
                vq_codes_pred,dur_pred = self.sd_lm.infer(txt_tokens, ph_vqcode, indur, spk_embed, txt_tokens_prompt.shape[1], ret,
                                    note=note,note_dur=note_dur,note_type=note_type)
                style = self.style_encoder.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
            
            # add dur
            dur = dur_pred
            src_padding = txt_tokens == 0
            dur = dur * (1 - src_padding.float())
            dur = (dur.exp() - 1).clamp(min=0)
            mel2ph = self.length_regulator(dur, src_padding).detach()
            mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
            style = expand_states(style.transpose(1, 2), mel2ph)

            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
            decoder_inp = expand_states(encoder_out, mel2ph)

            # add pitch embed
            midi_notes = None
            pitch_inp = (decoder_inp + spk_embed) * tgt_nonpadding
            f0, uv = None, None
            midi_notes = expand_states(note[:, :, None], mel2ph)
            
            if control:
                ret['content'] = decoder_inp
                ret['style'] = style
                ret['mel2ph'] = mel2ph_tgt = mel2ph
            else:
                ret['content'] = decoder_inp = decoder_inp[:, mel2ph_prompt.shape[1]:, :]
                ret['style'] = style = style[:,mel2ph_prompt.shape[1]:,:]
                ret['mel2ph'] = mel2ph_tgt = mel2ph[:,mel2ph_prompt.shape[1]:]
                midi_notes = midi_notes[:,mel2ph_prompt.shape[1]:,:]
                pitch_inp = pitch_inp[:,mel2ph_prompt.shape[1]:,:]
                tgt_nonpadding = tgt_nonpadding[:,mel2ph_prompt.shape[1]:,:]
            
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph_tgt, ret, midi_notes=midi_notes)

            # decoder input
            ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed + style) * tgt_nonpadding
            ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=True)
            return ret
    
