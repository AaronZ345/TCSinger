import os
import numpy as np
import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.text.text_encoder import build_token_encoder
from modules.TCSinger.tcsinger import SADecoder
from modules.TCSinger.sdlm import SDLM
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.audio import librosa_wav2spec
from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams as hp
from utils.audio.io import save_wav
import json


def process_align(ph_durs, mel, item, hop_size ,audio_sample_rate):
    mel2ph = np.zeros([mel.shape[0]], int)
    startTime = 0

    for i_ph in range(len(ph_durs)):
        start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
        mel2ph[start_frame:end_frame] = i_ph + 1
        startTime = startTime + ph_durs[i_ph]
    return mel2ph


class StyleControl(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = SDLM(dict_size, self.hparams)
        self.model_decoder=SADecoder()
        model.eval()
        load_ckpt(model, hparams['exp_name'], strict=False)
        load_ckpt(self.model_decoder, hparams['decoder_ckpt_dir'], strict=False)
        self.model_decoder.to(self.device)

        binary_data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_token_encoder(f'{binary_data_dir}/phone_set.json')
        return model

    def build_vocoder(self):
        vocoder = get_vocoder_cls(hparams["vocoder"])()
        return vocoder

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        notes, note_durs,note_types = sample["notes"], sample["note_durs"],sample['note_types']
        emotion, pace, range_,singing_method = sample['emotion'], sample['pace'], sample['range_'], sample['singing_method']
        mix_tech, falsetto_tech, breathy_tech, pharyngeal_tech, glissando_tech, vibrato_tech = sample['mix_tech'], sample['falsetto_tech'], sample['breathy_tech'], sample['pharyngeal_tech'], sample['glissando_tech'], sample['vibrato_tech']
        mel_prompt = sample['mel_prompt']
        
        output = {}

        # Run model
        with torch.no_grad():
            output = self.model.infer(txt_tokens, None,
                tgt_mels=None, 
                infer=True,
                mel_prompt=mel_prompt,
                note=notes, note_dur=note_durs, note_type=note_types,
                emotion=emotion,singing_method=singing_method,pace=pace,range_=range_,
                mix=mix_tech, falsetto=falsetto_tech, breathy=breathy_tech, pharyngeal=pharyngeal_tech, glissando=glissando_tech, vibrato=vibrato_tech,control=True)
            self.model_decoder(tgt_mels=None, infer=True, ret=output, spk_embed=None)

            # Get gen mel
            mel_out =  output['mel_out'][0]
            pred_f0 = output.get('f0_denorm_pred')[0]
            wav_out = self.vocoder.spec2wav(mel_out.cpu(),f0=pred_f0.cpu())

        wav_out = wav_out
        return wav_out, mel_out
    

    def preprocess_input(self, inp):
        ph=' '.join(inp['text_gen'])
        ph_token = self.ph_encoder.encode(ph)
        note= inp['note_gen']
        note_dur=inp['note_dur_gen']
        note_type=inp['note_type_gen']

        wav_fn=inp['ref_audio']
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        mel = np.clip(mel, hparams['mel_vmin'], hparams['mel_vmax'])
        
        tech_list=inp['tech']

        mix=[]
        falsetto=[]
        breathy=[]
        pharyngeal=[]
        vibrato=[]
        glissando=[]    
        for element in tech_list:
            mix.append(1 if '1' in element else 0)
            falsetto.append(1 if '2' in element else 0)
            breathy.append(1 if '3' in element else 0)
            pharyngeal.append(1 if '4' in element else 0)
            vibrato.append(1 if '5' in element else 0)
            glissando.append(1 if '6' in element else 0)

        item = {'item_name': wav_fn, 'text': ph, 'ph': ph,
                'ph_token': ph_token, 'mel': mel, 'note':note, 'note_dur':note_dur, 'note_type':note_type,
                'mix_tech': mix, 'falsetto_tech': falsetto,
                'breathy_tech': breathy, 'pharyngeal_tech': pharyngeal,
                'glissando_tech': glissando, 'vibrato_tech': vibrato,
                'emotion': inp['emotion'], 'pace': inp['pace'], 'range': inp['range_'],'singing_method':inp['singing_method']}
        item['ph_len'] = len(item['ph_token'])

        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'][:hparams['max_input_tokens']]).to(self.device)

        mix_tech = torch.LongTensor(item['mix_tech'])[None, :].to(self.device)
        falsetto_tech = torch.LongTensor(item['falsetto_tech'])[None, :].to(self.device)
        breathy_tech = torch.LongTensor(item['breathy_tech'])[None, :].to(self.device)
        pharyngeal_tech = torch.LongTensor(item['pharyngeal_tech'])[None, :].to(self.device)
        glissando_tech = torch.LongTensor(item['glissando_tech'])[None, :].to(self.device)
        vibrato_tech = torch.LongTensor(item['vibrato_tech'])[None, :].to(self.device)
        emotion = [item['emotion']]
        pace = [item['pace']]
        range_ = [item['range']]
        singing_method = [item['singing_method']]
        mels = torch.Tensor(item['mel'])[None, :].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'notes': note,
            'note_durs': note_dur,
            'note_types': note_type,
            'mix_tech': mix_tech,
            'falsetto_tech': falsetto_tech,
            'breathy_tech': breathy_tech,
            'vibrato_tech': vibrato_tech,
            'pharyngeal_tech': pharyngeal_tech,
            'glissando_tech': glissando_tech,
            'emotion': emotion,
            'pace': pace,
            'range_': range_,
            'singing_method': singing_method,
            'mel_prompt': mels,
        }
        return batch

    @classmethod
    def example_run(cls):

        set_hparams()
        
        inp = {
            # 'text_gen': ,
            # 'note_gen': ,
            # 'note_dur_gen' : ,
            # 'note_type_gen':,
            # 'ref_audio': ,
            'tech': [
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",
                "3","3",            
                ],
            'singing_method' : 'pop',
            'emotion': 'sad',
            'pace': 'slow',
            'range_': 'high',
            'ref_name': "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好#Control_Group#0000",
            'gen_name': "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好#Control_Group#0000",
        }
        
        # use info in metadata.json
        if 'ref_name' in inp:
            items_list = json.load(open(f"{hparams['processed_data_dir']}/metadata.json"))
            for item in items_list:
                if inp['ref_name'] in item['item_name']:
                    inp['ref_audio']=item['wav_fn']
                    break
            for item in items_list:        
                if inp['gen_name'] in item['item_name']:
                    inp['text_gen']=item['ph']
                    inp['note_gen']=item['ep_pitches']
                    inp['note_dur_gen'] =item['ep_notedurs']
                    inp['note_type_gen']=item['ep_types']    
                    # inp['tech']=item['tech']
                    # inp['singing_method']=item['singing_method']
                    # inp['emotion']=item['emotion']
                    # inp['pace']=item['pace']
                    # inp['range_']=item['range']
                    break         

        infer_ins = cls(hp)
        out = infer_ins.infer_once(inp)
        wav_out, mel_out = out
        os.makedirs('infer_out', exist_ok=True)
        save_wav(wav_out, f'infer_out/control.wav', hp['audio_sample_rate'])


if __name__ == '__main__':
    StyleControl.example_run()
