# TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control

#### Yu Zhang, Ziyue Jiang, Ruiqi Li, Changhao Pan, Jinzheng He, Rongjie Huang, Chuxin Wang, Zhou Zhao | Zhejiang University

PyTorch Implementation of [TCSinger (EMNLP 2024)](https://arxiv.org/abs/2409.15977): Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.15977)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/777601485)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/TCSinger?style=social)](https://github.com/AaronZ345/TCSinger)

We provide our implementation and pre-trained models in this repository.

Visit our [demo page](https://tcsinger.github.io/) for audio samples.

## News
- 2024.09: We released the full dataset of [GTSinger](https://github.com/GTSinger)!
- 2024.09: TCSinger is accepted by EMNLP 2024!

## Key Features
- We present **TCSinger**, the first zero-shot SVS model for style transfer across cross-lingual speech and singing styles, along with multi-level style control. TCSinger excels in personalized and controllable SVS tasks.
- We introduce the **clustering style encoder** to extract styles, and the **Style and Duration Language Model (S&D-LM)** to predict both style information and phoneme duration, addressing style modeling, transfer, and control.
- We propose the **style adaptive decoder** to generate intricately detailed songs using a novel mel-style adaptive normalization method.
- Experimental results show that TCSinger surpasses baseline models in synthesis quality, singer similarity, and style controllability across various tasks: **zero-shot style transfer, multi-level style control, cross-lingual style transfer, and speech-to-singing style transfer**.

## Quick Start
We provide an example of how you can generate high-fidelity samples using TCSinger.

To try on your own dataset or GTSinger, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

**The code will come soon...**

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[StyleSinger](https://github.com/AaronZ345/StyleSinger)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{zhang2024tcsinger,
  title={TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control},
  author={Zhang, Yu and Jiang, Ziyue and Li, Ruiqi and Pan, Changhao and He, Jinzheng and Huang, Rongjie and Wang, Chuxin and Zhao, Zhou},
  journal={arXiv preprint arXiv:2409.15977},
  year={2024}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/TCSinger)
