from modules.TCSinger.tcsinger import TCSinger, SADecoder
from tasks.TCSinger.base_gen_task import AuxDecoderMIDITask
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.TCSinger.dataset import TCDataset


class TCSingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = TCDataset
        self.mse_loss_fn = torch.nn.MSELoss()

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = TCSinger(dict_size, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        ph_lengths = sample['txt_lengths']
        if hparams['use_spk_embed']==True:
            spk_embed=sample['spk_embed']
        else:
            spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        mel_prompt = sample["mel_prompt"]

        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=None,target=target,ph_lengths=ph_lengths, f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types, mel_prompt=mel_prompt)
        losses = {}
        
        self.add_mel_loss(output['mel_out'], target, losses)

        if 'gdiff' in output:
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]

        if 'vq_loss' in output:
            losses['vq_loss']=output['vq_loss']

        return losses, output

# task with sad
class SADTask(TCSingerTask):
    def __init__(self):
        super(SADTask, self).__init__()

    def build_model(self):
        self.build_pretrain_model()
        self.model = SADecoder()

    def build_pretrain_model(self):
        dict_size = len(self.token_encoder)
        self.pretrain = TCSinger(dict_size, hparams)
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        ph_lengths = sample['txt_lengths']
        if hparams['use_spk_embed']==True:
            spk_embed=sample['spk_embed']
        else:
            spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        mel_prompt = sample["mel_prompt"]

        output = self.pretrain(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=None,target=target,ph_lengths=ph_lengths, f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types, mel_prompt=mel_prompt)

        self.model(target, infer, output, spk_embed)
        losses = {}
        losses["diff"] = output["diff"]
        return losses, output
    
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output
    
    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])
