from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
import random
from utils.commons.indexed_datasets import IndexedDataset
from tqdm import tqdm
import numpy as np


# single data
class TCDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.get_spk_prompt()

    # random choose spk prompt
    def get_spk_prompt(self):
        self.spkid2idx = {}
        temp_indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')

        for idx in tqdm(range(len(self)), total=len(self)):
            item = temp_indexed_ds[self.avail_idxs[idx]]
            spk_id = int(item['spk_id'])
            
            if spk_id not in self.spkid2idx:
                self.spkid2idx[spk_id] =[idx]
            else:
                self.spkid2idx[spk_id].append(idx)

    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(TCDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

        random_number = random.random()

        # Extract emotion, tech, singing_method, pace, range if emotion exists
        if 'emotion' in item and random_number > 0.1:
            sample['emotion'] = item['emotion']
            sample['singing_method'] = item['singing_method']
            sample['pace'] = item['pace']
            sample['range'] = item['range']

        spk_id = int(item['spk_id'])
        prompt_index = random.choice(self.spkid2idx[spk_id])
        prompt_item = self._get_item(prompt_index)
        assert len(prompt_item['mel']) == self.sizes[prompt_index], (len(prompt_item['mel']), self.sizes[prompt_index])
        max_frames = hparams['max_prompt_frames']
        spec = torch.Tensor(prompt_item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']

        mel2ph_len = sum((np.array(prompt_item["mel2ph"]) > 0).astype(np.int64))
        T = min(max_frames, mel2ph_len, len(prompt_item["f0"]))
        sample['mel_prompt'] = spec[:T]

        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(TCDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types
        batch['mel_prompt'] = collate_1d_or_2d([s['mel_prompt'] for s in samples], 0)

        return batch


# target with prompt
class SDLMDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        mel = item['mel']
        
        assert len(mel) == self.sizes[index], (len(mel), self.sizes[index])
        max_frames = hparams['max_frames']
        mel = torch.Tensor(mel)[:max_frames]
        max_frames = mel.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        mel = mel[:max_frames]
        ph_token = torch.LongTensor(item['ph_token'][:hparams['max_input_tokens']])
        T = mel.shape[0]
        mel2ph = torch.LongTensor(item['mel2ph'])[:T]

        sample = {}
        if index == 0 or index + 1 == len(self.avail_idxs):
            sample['avail_flag'] = False
        else:
            item_prev = self._get_item(index - 1)
            item_next = self._get_item(index + 1)

            if (item_prev['spk_id'] != item['spk_id']) and (item_next['spk_id'] != item['spk_id']):
                sample['avail_flag'] = False
            else:
                # Load prompt
                sample['avail_flag'] = True
                if item_prev['spk_id'] == item['spk_id']:
                    item_prompt = item_prev
                    index_prompt = index - 1
                else:
                    item_prompt = item_next
                    index_prompt = index + 1
                mel_prompt = item_prompt['mel']
                assert len(mel_prompt) == self.sizes[index_prompt], (len(mel_prompt), self.sizes[index_prompt])
                max_frames = hparams['max_frames']
                mel_prompt = torch.Tensor(mel_prompt)[:max_frames]
                max_frames = mel_prompt.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
                mel_prompt = mel_prompt[:max_frames]
                mel2ph_prompt = torch.LongTensor(item_prompt['mel2ph'])[:mel_prompt.shape[0]]
                txt_token_prompt = torch.LongTensor(item_prompt['ph_token'][:hparams['max_input_tokens']])

                note_prompt = torch.LongTensor(item_prompt['ep_pitches'][:hparams['max_input_tokens']])
                note_dur_prompt = torch.FloatTensor(item_prompt['ep_notedurs'][:hparams['max_input_tokens']])
                note_type_prompt = torch.LongTensor(item_prompt['ep_types'][:hparams['max_input_tokens']])
                sample["note_prompt"], sample["note_dur_prompt"], sample["note_type_prompt"] = note_prompt, note_dur_prompt, note_type_prompt

                note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
                note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
                note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
                sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

                random_number = random.random()

                # Extract emotion, singing_method, pace, range if emotion exists
                if 'emotion' in item and random_number > 0.1:
                    sample['emotion'] = item['emotion']
                    sample['singing_method'] = item['singing_method']
                    sample['pace'] = item['pace']
                    sample['range'] = item['range']

                    # Process tech features for current item
                    mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
                    falsetto = torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
                    breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
                    pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
                    glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])
                    vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
                    sample['mix'], sample['falsetto'], sample['breathy'], sample['pharyngeal'], sample['glissando'], sample['vibrato'] = mix, falsetto, breathy, pharyngeal, glissando, vibrato

                    # Process tech features for prompt item
                    mix_prompt = torch.LongTensor(item_prompt['mix_tech'][:hparams['max_input_tokens']])
                    falsetto_prompt = torch.LongTensor(item_prompt['falsetto_tech'][:hparams['max_input_tokens']])
                    breathy_prompt = torch.LongTensor(item_prompt['breathy_tech'][:hparams['max_input_tokens']])
                    pharyngeal_prompt = torch.LongTensor(item_prompt['pharyngeal_tech'][:hparams['max_input_tokens']])
                    glissando_prompt = torch.LongTensor(item_prompt['glissando_tech'][:hparams['max_input_tokens']])
                    vibrato_prompt = torch.LongTensor(item_prompt['vibrato_tech'][:hparams['max_input_tokens']])
                    sample['mix_prompt'], sample['falsetto_prompt'], sample['breathy_prompt'], sample['pharyngeal_prompt'], sample['glissando_prompt'], sample['vibrato_prompt'] = mix_prompt, falsetto_prompt, breathy_prompt, pharyngeal_prompt, glissando_prompt, vibrato_prompt
                else:
                    sample['emotion'] = 'no'
                    sample['singing_method'] = 'no'
                    sample['pace'] = 'no'
                    sample['range'] = 'no'
                    seq_length = len(sample['note'])
                    seq_length_prompt = len(sample['note_prompt'])
                
                    # Create sequences of length `seq_length` and `seq_length_prompt` filled with 2
                    mix = torch.LongTensor([2] * seq_length)
                    falsetto = torch.LongTensor([2] * seq_length)
                    breathy = torch.LongTensor([2] * seq_length)
                    mix_prompt = torch.LongTensor([2] * seq_length_prompt)
                    falsetto_prompt = torch.LongTensor([2] * seq_length_prompt)
                    breathy_prompt = torch.LongTensor([2] * seq_length_prompt)

                    pharyngeal = torch.LongTensor([2] * seq_length)
                    glissando = torch.LongTensor([2] * seq_length)
                    vibrato = torch.LongTensor([2] * seq_length)
                    sample['mix'], sample['falsetto'], sample['breathy'], sample['pharyngeal'], sample['glissando'], sample['vibrato'] = mix, falsetto, breathy, pharyngeal, glissando, vibrato

                    pharyngeal_prompt = torch.LongTensor([2] * seq_length_prompt)
                    glissando_prompt = torch.LongTensor([2] * seq_length_prompt)
                    vibrato_prompt = torch.LongTensor([2] * seq_length_prompt)
                    sample['mix_prompt'], sample['falsetto_prompt'], sample['breathy_prompt'], sample['pharyngeal_prompt'], sample['glissando_prompt'], sample['vibrato_prompt'] = mix_prompt, falsetto_prompt, breathy_prompt, pharyngeal_prompt, glissando_prompt, vibrato_prompt

                if hparams['use_pitch_embed']:
                    assert 'f0' in item
                    f0, uv = norm_interp_f0(item["f0"][:T])
                    uv = torch.FloatTensor(uv)
                    f0 = torch.FloatTensor(f0)

                    f0_prompt, uv_prompt = norm_interp_f0(item_prompt["f0"][:mel_prompt.shape[0]])
                    uv_prompt = torch.FloatTensor(uv_prompt)
                    f0_prompt = torch.FloatTensor(f0_prompt)

                    if hparams['pitch_type'] == 'ph':
                        if "f0_ph" in item:
                            f0 = torch.FloatTensor(item['f0_ph'])
                        else:
                            f0 = denorm_f0(f0, None)
                        f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                        f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                            0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                        f0_ph = f0_phlevel_sum / f0_phlevel_num
                        f0, uv = norm_interp_f0(f0_ph)
                else:
                    f0, uv = None, None
                    f0_prompt, uv_prompt = None, None
                sample["f0"], sample["uv"] = f0, uv
                sample["f0_prompt"], sample["uv_prompt"] = f0_prompt, uv_prompt
                if hparams['use_spk_embed']:
                    sample["spk_embed"] = torch.Tensor(item['spk_embed'])
                if hparams['use_spk_id']:
                    sample["spk_id"] = int(item['spk_id'])

                sample.update({
                    "id": index,
                    "item_name": item['item_name'],
                    "text": item['ph'],
                    'text_prompt': item_prompt['ph'],
                    "txt_token": ph_token,
                    "txt_token_prompt": txt_token_prompt,
                    "mel": mel,
                    "mel_prompt": mel_prompt,
                    "mel_nonpadding": mel.abs().sum(-1) > 0,
                    "mel2ph": mel2ph,
                    "mel2ph_prompt": mel2ph_prompt,
                })
        return sample

    def collater(self, samples):
        # Skip some failed cases
        if len(samples) == 0:
            return {}
        all_failed = True
        for s in samples:
            if s['avail_flag']:
                all_failed = False
        if all_failed:
            return {}

        # Collate current sentence
        hparams = self.hparams
        txt_tokens_list = []
        mels_list = []
        mel2ph_list = []
        spk_embed_prompt_list = []
        note_list = []
        note_dur_list = []
        note_type_list = []
        f0_list = []
        uv_list = []
        spk_list = []
        emotion_list = []
        singing_method_list = []
        pace_list = []
        range_list = []
        mix_list = []
        falsetto_list = []
        breathy_list = []
        pharyngeal_list = []
        glissando_list = []
        vibrato_list = []
        for idx, item in enumerate(samples):
            if item['avail_flag']:
                txt_token = torch.cat((item['txt_token_prompt'], item['txt_token']), dim=0)
                note = torch.cat((item['note_prompt'], item['note']), dim=0)
                note_dur = torch.cat((item['note_dur_prompt'], item['note_dur']), dim=0)
                note_type = torch.cat((item['note_type_prompt'], item['note_type']), dim=0)
                mel = torch.cat((item['mel_prompt'], item['mel']), dim=0)
                f0 = torch.cat((item['f0_prompt'], item['f0']), dim=0)
                uv = torch.cat((item['uv_prompt'], item['uv']), dim=0)
                mel2ph_shift = item['txt_token_prompt'].shape[0] - item['mel2ph_prompt'].max()
                mel2ph = torch.cat((item['mel2ph_prompt'],
                                    (item['mel2ph_prompt'].max() + item['mel2ph'] + mel2ph_shift)), dim=0)
                
                # Process tech features
                mix = torch.cat((item['mix_prompt'], item['mix']), dim=0)
                falsetto = torch.cat((item['falsetto_prompt'], item['falsetto']), dim=0)
                breathy = torch.cat((item['breathy_prompt'], item['breathy']), dim=0)
                pharyngeal = torch.cat((item['pharyngeal_prompt'], item['pharyngeal']), dim=0)
                glissando = torch.cat((item['glissando_prompt'], item['glissando']), dim=0)
                vibrato = torch.cat((item['vibrato_prompt'], item['vibrato']), dim=0)

                note_list.append(note)
                note_dur_list.append(note_dur)
                note_type_list.append(note_type)
                f0_list.append(f0)
                uv_list.append(uv)
                txt_tokens_list.append(txt_token)
                mels_list.append(mel)
                mel2ph_list.append(mel2ph)
                mix_list.append(mix)
                falsetto_list.append(falsetto)
                breathy_list.append(breathy)
                pharyngeal_list.append(pharyngeal)
                glissando_list.append(glissando)
                vibrato_list.append(vibrato)

                # Add emotion and other fields if available
                emotion_list.append(item['emotion'])
                singing_method_list.append(item.get('singing_method', None))
                pace_list.append(item.get('pace', None))
                range_list.append(item.get('range', None))

                if hparams['use_spk_embed']:
                    spk_embed_prompt_list.append(item['spk_embed'])
                if hparams['use_spk_id']:
                    spk_ids = item['spk_id']
                    spk_list.append(spk_ids)

        item_names = [s['item_name'] for s in samples if s['avail_flag']]
        text = [s['text'] for s in samples if s['avail_flag']]
        txt_tokens = collate_1d_or_2d(txt_tokens_list, 0)
        mels = collate_1d_or_2d(mels_list, 0.0)
        mel2ph = collate_1d_or_2d(mel2ph_list, 0.0)
        note = collate_1d_or_2d(note_list, 0.0)
        note_dur = collate_1d_or_2d(note_dur_list, 0.0)
        note_type = collate_1d_or_2d(note_type_list, 0.0)

        # Collate prompts
        txt_tokens_gen = collate_1d_or_2d([s['txt_token'] for s in samples if s['avail_flag']], 0)
        txt_tokens_prompt = collate_1d_or_2d([s['txt_token_prompt'] for s in samples if s['avail_flag']], 0)
        mel_prompt = collate_1d_or_2d([s['mel_prompt'] for s in samples if s['avail_flag']], 0)
        mel2ph_prompt = collate_1d_or_2d([s['mel2ph_prompt'] for s in samples if s['avail_flag']], 0)
        note_prompt = collate_1d_or_2d([s['note_prompt'] for s in samples if s['avail_flag']], 0.0)
        note_dur_prompt = collate_1d_or_2d([s['note_dur_prompt'] for s in samples if s['avail_flag']], 0.0)
        note_type_prompt = collate_1d_or_2d([s['note_type_prompt'] for s in samples if s['avail_flag']], 0.0)
        text_prompt = [s['text_prompt'] for s in samples if s['avail_flag']]

        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d(f0_list, 0.0)
            uv = collate_1d_or_2d(uv_list, 0.0)
        else:
            f0, uv = None, None

        # Collate tech features
        mix = collate_1d_or_2d(mix_list, 0)
        falsetto = collate_1d_or_2d(falsetto_list, 0)
        breathy = collate_1d_or_2d(breathy_list, 0)
        pharyngeal = collate_1d_or_2d(pharyngeal_list, 0)
        glissando = collate_1d_or_2d(glissando_list, 0)
        vibrato = collate_1d_or_2d(vibrato_list, 0)

        # Collate the additional fields (emotion, singing_method, pace, range)
        emotion = emotion_list
        singing_method = singing_method_list
        pace = pace_list
        range_ = range_list

        batch = {
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'text_prompt': text_prompt,
            'txt_tokens': txt_tokens,
            'txt_tokens_gen': txt_tokens_gen,
            'txt_tokens_prompt': txt_tokens_prompt,
            'mels': mels,
            'mel_prompt': mel_prompt,
            'mel2ph': mel2ph,
            'mel2ph_prompt': mel2ph_prompt,
            'f0': f0,
            'uv': uv,
        }
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor(spk_list)
            batch['spk_id'] = spk_ids

        batch["notes"], batch["note_durs"], batch["note_types"] = note, note_dur, note_type
        batch["notes_prompt"], batch["note_durs_prompt"], batch["note_types_prompt"] = note_prompt, note_dur_prompt, note_type_prompt

        # Include emotion and related fields in the batch
        batch["emotion"] = emotion
        batch["singing_method"] = singing_method
        batch["pace"] = pace
        batch["range"] = range_
        batch['mix'], batch['falsetto'], batch['breathy'], batch['pharyngeal'], batch['glissando'], batch['vibrato'] = mix, falsetto, breathy, pharyngeal, glissando, vibrato

        if hparams['use_spk_embed']:
            batch['spk_embed_prompt'] = torch.stack(spk_embed_prompt_list)
        return batch
