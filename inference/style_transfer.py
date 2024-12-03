import os
import numpy as np
import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.text.text_encoder import build_token_encoder
from tasks.TCSinger.base_gen_task import mel2ph_to_dur
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


class StyleTransfer(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = SDLM(dict_size, self.hparams)
        self.model_post=SADecoder()
        model.eval()
        load_ckpt(model, hparams['exp_name'], strict=False)
        load_ckpt(self.model_post, hparams['post_ckpt_dir'], strict=False)
        self.model_post.to(self.device)

        binary_data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_token_encoder(f'{binary_data_dir}/phone_set.json')
        return model

    def build_vocoder(self):
        vocoder = get_vocoder_cls(hparams["vocoder"])()
        return vocoder

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        txt_tokens_prompt = sample['txt_tokens_prompt']
        mel2ph_prompt = sample['mel2ph_prompt']
        mel_prompt = sample['mel_prompt']
        notes, note_durs,note_types = sample["notes"], sample["note_durs"],sample['note_types']
        output = {}
            
        B, T = txt_tokens_prompt.shape
        nonpadding = (txt_tokens_prompt != 0).float()
        ref_dur = mel2ph_to_dur(mel2ph_prompt, T).float() * nonpadding
        ref_dur=(ref_dur + 1).log()

        # Run model
        with torch.no_grad():
            output = self.model.infer(txt_tokens, 
                txt_tokens_prompt,
                tgt_mels=None, 
                mel_prompt=mel_prompt,
                mel2ph_prompt=mel2ph_prompt,
                ref_dur=ref_dur,
                infer=True,
                note=notes, note_dur=note_durs, note_type=note_types, control=False)
            self.model_post(tgt_mels=None, infer=True, ret=output, spk_embed=None)

            # Get gen mel
            mel_out =  output['mel_out'][0]
            pred_f0 = output.get('f0_denorm_pred')[0]
            wav_out = self.vocoder.spec2wav(mel_out.cpu(),f0=pred_f0.cpu())

        wav_out = wav_out
        return wav_out, mel_out
    

    def preprocess_input(self, inp):
        ph=' '.join(inp['text_in'])
        ph_token_prompt = self.ph_encoder.encode(ph)
        sent_txt_lengths = [len(ph_token_prompt)]
        ph_gen=' '.join(inp['text_gen'])
        ph_token_gen = self.ph_encoder.encode(ph_gen)
        ph_token = ph_token_prompt + ph_token_gen
        note=inp['note_in'] + inp['note_gen']
        note_dur=inp['note_dur_in'] + inp['note_dur_gen']
        note_type=inp['note_type_in'] + inp['note_type_gen']

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

        item = {'item_name': wav_fn, 'text': ph, 'ph': ph,
                'ph_token': ph_token, 'ph_token_gen': ph_token_gen, 'ph_token_prompt': ph_token_prompt,
                'mel': mel, 'note':note, 'note_dur':note_dur,'note_type':note_type,
                'sent_txt_lengths': sent_txt_lengths}
        
        item['mel2ph_prompt'] = process_align(inp["ph_durs"], mel, item,hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate'])

        item['ph_len'] = seq_length = len(item['ph_token'])

        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_tokens_gen = torch.LongTensor(item['ph_token_gen'])[None, :].to(self.device)
        txt_tokens_prompt = torch.LongTensor(item['ph_token_prompt'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        mels = torch.Tensor(item['mel'])[None, :].to(self.device)
        mel2ph_prompt = torch.LongTensor(item['mel2ph_prompt'])[None, :].to(self.device)
        mel_lengths = torch.LongTensor([mels.shape[1]]).to(self.device)
        sent_txt_lengths = torch.LongTensor(item['sent_txt_lengths'])[None, :].to(self.device)
        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'][:hparams['max_input_tokens']]).to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_tokens_gen': txt_tokens_gen,
            'txt_tokens_prompt': txt_tokens_prompt,
            'txt_lengths': txt_lengths,
            'sent_txt_lengths': sent_txt_lengths,
            'mel_prompt': mels,
            'mel2ph_prompt': mel2ph_prompt,
            'mel_lengths': mel_lengths,
            'notes': note,
            'note_durs': note_dur,
            'note_types': note_type
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
            # 'text_in': ,
            # 'note_in': ,
            # 'note_dur_in' :,
            # 'note_type_in':,
            # 'ref_audio': ,
            # 'ph_durs':
            'ref_name':"English#EN-Alto-2#Mixed_Voice_and_Falsetto#A Thousand Years#Control_Group#0002",
            'gen_name': "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好#Mixed_Voice_Group#0001",
        }
        
        # use info in metadata.json
        if 'ref_name' in inp:
            items_list = json.load(open(f"{hparams['processed_data_dir']}/metadata.json"))
            for item in items_list:
                if inp['ref_name'] in item['item_name']:
                    inp['text_in']=item['ph']
                    inp['note_in']=item['ep_pitches']
                    inp['note_dur_in'] =item['ep_notedurs']
                    inp['note_type_in']=item['ep_types']
                    inp['ref_audio']=item['wav_fn'].replace('/home2/zhangyu/data/nips_final/nips_submit','/root/autodl-tmp/data/singing/GTSinger')
                    inp['ph_durs']=item['ph_durs']
                    break
            for item in items_list:        
                if inp['gen_name'] in item['item_name']:
                    inp['text_gen']=item['ph']
                    inp['note_gen']=item['ep_pitches']
                    inp['note_dur_gen'] =item['ep_notedurs']
                    inp['note_type_gen']=item['ep_types']    
                    break         

        infer_ins = cls(hp)
        out = infer_ins.infer_once(inp)
        wav_out, mel_out = out
        os.makedirs('infer_out', exist_ok=True)
        save_wav(wav_out, f'infer_out/transfer.wav', hp['audio_sample_rate'])


if __name__ == '__main__':
    StyleTransfer.example_run()
