# TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control

#### Yu Zhang, Ziyue Jiang, Ruiqi Li, Changhao Pan, Jinzheng He, Rongjie Huang, Chuxin Wang, Zhou Zhao | Zhejiang University

PyTorch Implementation of [TCSinger (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.117/): Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.15977)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/777601485)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model)](https://huggingface.co/AaronZ345/TCSinger)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/TCSinger?style=social)](https://github.com/AaronZ345/TCSinger)

We provide our implementation and pre-trained models in this repository.

Visit our [demo page](https://aaronz345.github.io/TCSingerDemo/) for audio samples.

## News
- 2024.12: We released the checkpoints of TCSinger!
- 2024.11: We released the code of TCSinger!
- 2024.09: We released the full dataset of [GTSinger](https://github.com/GTSinger)!
- 2024.09: TCSinger is accepted by EMNLP 2024!

## Key Features
- We present **TCSinger**, the first zero-shot SVS model for style transfer across cross-lingual speech and singing styles, along with multi-level style control. TCSinger excels in personalized and controllable SVS tasks.
- We introduce the **clustering style encoder** to extract styles, and the **Style and Duration Language Model (S&D-LM)** to predict both style information and phoneme duration, addressing style modeling, transfer, and control.
- We propose the **style adaptive decoder** to generate intricately detailed songs using a novel mel-style adaptive normalization method.
- Experimental results show that TCSinger surpasses baseline models in synthesis quality, singer similarity, and style controllability across various tasks: **zero-shot style transfer, multi-level style control, cross-lingual style transfer, and speech-to-singing style transfer**.

## Quick Start
We provide an example of how you can generate high-fidelity samples using TCSinger.

To try on your own dataset or GTSinger, simply clone this repo on your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

### Pre-trained Models
You can use all pre-trained models we provide on [HuggingFace](https://huggingface.co/AaronZ345/TCSinger) or [Google Drive](https://drive.google.com/drive/folders/1t57KKccSMGkrJhCRRCTo6XoXhCmZHFxl?usp=drive_link). **Notably, this TCSinger checkpoint only supports Chinese and English! You should train your own model based on GTSinger for multilingual style transfer and control!** Details of each folder are as follows:

| Model       |  Description                                                              | 
|-------------|--------------------------------------------------------------------------|
| TCSinger |  Acousitic model [(config)](./egs/tcsinger.yaml) |
| SAD |  Decoder model [(config)](./egs/sad.yaml) |
| SDLM |  LM model [(config)](./egs/sdlm.yaml) |
| HIFI-GAN    |  Neural Vocoder               |

### Dependencies

A suitable [conda](https://conda.io/) environment named `tcsinger` can be created
and activated with:

```
conda create -n tcsinger python=3.10
conda install --yes --file requirements.txt
conda activate tcsinger
```

### Multi-GPU

By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Inference for bilingual singing voices

Here we provide a speech synthesis pipeline using TCSinger.

1. Prepare **TCSinger, SAD, SDLM**: Download and put checkpoint at `checkpoints/TCSinger`, `checkpoints/SAD`, `checkpoints/SDLM`.
2. Prepare **HIFI-GAN**: Download and put checkpoint at `checkpoints/hifigan`.
3. Prepare **prompt information**: Provide a prompt_audio (48k) and input target ph, target note for each ph, target note_dur for each ph, target note_type for each ph (rest: 1, lyric: 2, slur: 3), and prompt audio path, prompt ph, prompt note, note_dur, note_type. Input these information in `Inference/style_transfer.py`. **Notably, if you want to use Chinese and English data in GTSinger to infer this checkpoint, refer to [phone_set](./ZHEN_checkpoint_phone_set.json), you have to delete _zh or _en in each ph of GTSinger!**
4. Infer with tcsinger with style transfer:

```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/style_transfer.py --config egs/sdlm.yaml  --exp_name checkpoints/SDLM
```

or 

3. Prepare **prompt information**: Provide a prompt_audio (48k) and input target ph, target note for each ph, target note_dur for each ph, target note_type for each ph (rest: 1, lyric: 2, slur: 3), and style information. Input these information in `Inference/style_control.py`. **Notably, if you want to use Chinese and English data in GTSinger to infer this checkpoint, refer to [phone_set](./ZHEN_checkpoint_phone_set.json), you have to delete _zh or _en in each ph of GTSinger!**
4. Infer with tcsinger with style control **(the effectiveness of the style_control feature is suboptimal for certain timbres due to the inclusion of speech and unannotated data. I recommend using GTSinger or other datasets for fine-tuning before style control inference.)**:

```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/style_control.py --config egs/sdlm.yaml  --exp_name checkpoints/SDLM
```

Generated wav files are saved in `infer_out` by default.<br>

## Train your own model based on GTSinger

### Data Preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/AaronZ345/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/tc` **(Note: we provide `metadata.json` and `phone_set.json` in GTSinger, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/tc`), `binary_data_dir`,`valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](./egs/TCSinger.yaml).
4. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/runs/binarize.py --config egs/TCSinger.yaml
```

### Training TCSinger

1. Train Main Model:
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/tcsinger.yaml  --exp_name TCSinger --reset
```
2. Train SAD:
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/sad.yaml  --exp_name SAD --reset
```
3. Train SDLM:
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/sdlm.yaml  --exp_name SDLM --reset
```

### Inference with TCSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/sdlm.yaml  --exp_name SDLM --infer
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[StyleSinger](https://github.com/AaronZ345/StyleSinger)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@inproceedings{zhang2024tcsinger,
  title={TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control},
  author={Zhang, Yu and Jiang, Ziyue and Li, Ruiqi and Pan, Changhao and He, Jinzheng and Huang, Rongjie and Wang, Chuxin and Zhao, Zhou},
  booktitle={EMNLP},
  year={2024}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/TCSinger)
