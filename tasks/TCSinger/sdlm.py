from modules.TCSinger.tcsinger import TCSinger, SADecoder
from tasks.TCSinger.base_gen_task import AuxDecoderMIDITask,f0_to_figure,mel2ph_to_dur
from utils.commons.hparams import hparams
import torch
import torch.nn.functional as F
from utils.commons.ckpt_utils import load_ckpt
from tasks.TCSinger.dataset import SDLMDataset
from utils.nn.model_utils import print_arch
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.commons.multiprocess_utils import MultiprocessManager
import os
from modules.TCSinger.sdlm import SDLM


# final task with sdlm
class SDLMTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = SDLMDataset
        self.mse_loss_fn = torch.nn.MSELoss()

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = SDLM(dict_size, hparams)
        self.model_ = TCSinger(dict_size, hparams)
        self.model_decoder = SADecoder()

    def build_model(self):
        self.build_tts_model()
        load_ckpt(self.model_, hparams['fs2_ckpt_dir'], strict=False)
        load_ckpt(self.model_decoder, hparams['decoder_ckpt_dir'], strict=False)
        
        # Copy params
        src_state_dict = self.model_.state_dict()
        dest_state_dict = self.model.state_dict()
        for name in dest_state_dict:
            if 'sd_lm' not in name:
                dest_state_dict[name].data.copy_(src_state_dict[name].data)
        self.model.load_state_dict(dest_state_dict)
        del self.model_
        del src_state_dict
        
        # Freeze some parameters in VQVAE_LM
        for item in self.model.named_parameters():
            name, param = item
            if 'sd_lm' not in name:
                param.requires_grad = False

        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
        print_arch(self.model)
        return self.model

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        txt_tokens_prompt = sample['txt_tokens_prompt']  # [B, T_t]
        mels = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        mel_prompt = sample['mel_prompt']
        mel2ph_prompt = sample['mel2ph_prompt']
        spk_embed_prompt = sample.get('spk_embed_prompt', None)
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        B, T = txt_tokens_prompt.shape
        nonpadding = (txt_tokens_prompt != 0).float()
        ref_dur = mel2ph_to_dur(mel2ph_prompt, T).float() * nonpadding
        ref_dur=(ref_dur + 1).log()
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        if not infer:
            tgt_dur = mel2ph_to_dur(mel2ph, T).float() * nonpadding
            tgt_dur = (tgt_dur + 1).log()
        else:
            tgt_dur = None

        mix,falsetto,breathy,pharyngeal,glissando,vibrato=sample['mix'],sample['falsetto'],sample['breathy'],sample['pharyngeal'],sample['glissando'],sample['vibrato']
        emotion,pace,range_,singing_method=sample['emotion'],sample['pace'],sample['range'],sample['singing_method']

        output = {}
        # Run model
        f0 = sample['f0']  # [B, T_s]
        uv = sample['uv']  # [B, T_s] 0/1

        output = self.model(txt_tokens, 
            txt_tokens_prompt,
            tgt_mels=mels, 
            mel2ph=mel2ph, 
            mel_prompt=mel_prompt,
            mel2ph_prompt=mel2ph_prompt,ref_dur=ref_dur,tgt_dur=tgt_dur,
            spk_embed_prompt=spk_embed_prompt,infer=infer,
            note=notes, note_dur=note_durs, note_type=note_types,
            f0=f0, uv=uv,
            mix=mix,falsetto=falsetto,breathy=breathy,pharyngeal=pharyngeal,glissando=glissando,vibrato=vibrato,
            emotion=emotion,pace=pace,range_=range_,singing_method=singing_method)
        # Add losses
        losses = {}
        if not infer:
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            losses['psd_pred'] = F.cross_entropy(
                output['vq_codes_pred'].transpose(1, 2), output['vq_codes'], ignore_index=0)
        return losses, output

    def validation_step(self, sample, batch_idx):
        if sample=={}:
            return {}
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], _ = self.run_model(sample, infer=False)
        _, model_out = self.run_model(sample, infer=True)
        self.model_decoder(tgt_mels=sample['mels'], infer=True, ret=model_out, spk_embed=None)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            
            gt_f0 = gt_f0[0][sample['mel2ph_prompt'].shape[1]:]
            gt_mel = sample['mels'][0][sample['mel2ph_prompt'].shape[1]:]
            
            wav_gt = self.vocoder.spec2wav(gt_mel[:len(gt_f0)].cpu().numpy(), f0=gt_f0[:len(gt_mel)].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0][:len(model_out['mel_out'][0])].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, gt_mel, model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0, None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.result_f0s_path = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}', "result_f0s.npy")
        self.result_f0s = []
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/wavs', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)
        if hparams.get('save_mel_npy', False):
            os.makedirs(f'{self.gen_dir}/mel_npy', exist_ok=True)

    def test_step(self, sample, batch_idx):
        if sample=={}:
            return {}
        mel2ph = sample['mel2ph']
        sample["mel2ph"] = None
        _, outputs = self.run_model(sample, infer=True)
        self.model_decoder(tgt_mels=sample['mels'], infer=True, ret=outputs, spk_embed=None)
        sample["mel2ph"] = mel2ph
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
        f0_pred = outputs.get('f0_denorm_pred')[0].cpu().numpy()
        self.result_f0s.append({"gt": f0, "pred": f0_pred})
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        mel2ph = sample["mel2ph"][0].cpu().numpy()
        mel2ph_pred = outputs.get("mel2ph")
        if mel2ph_pred is not None:
            mel2ph_pred = mel2ph_pred[0].cpu().numpy()
        base_fn = f'{item_name}[%s]'
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'Pred', gen_dir, None, None, None, None, None])
        
        mel_pt = mel_gt[:sample['mel2ph_prompt'].shape[1]]
        f0_pt = f0[:sample['mel2ph_prompt'].shape[1]]
        wav_pt = self.vocoder.spec2wav(mel_pt, f0=f0_pt)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pt, mel_pt, base_fn % 'Prompt', gen_dir, None, None, None, None, None])
        
        if hparams['save_gt']:
            mel_gt = mel_gt[sample['mel2ph_prompt'].shape[1]:]
            f0 = f0[sample['mel2ph_prompt'].shape[1]:]
            wav_gt = self.vocoder.spec2wav(mel_gt[:len(f0)], f0=f0[:len(mel_gt)])
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'GT', gen_dir, str_phs, mel2ph, f0, f0_pred, None])
        return {}
    
