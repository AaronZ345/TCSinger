[Back to TCSinger](../README.md)

# Train TCSinger on GTSinger

Run these commands from the repository root in the environment described in the main README. Set `GPU` to an existing device, for example `export GPU=0`.

## Data preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/AaronZ345/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/tc` **(Note: we provide `metadata.json` and `phone_set.json` in GTSinger, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/tc`), `binary_data_dir`,`valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](../egs/tcsinger.yaml).
4. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/runs/binarize.py --config egs/tcsinger.yaml
```

## Training

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

## Inference

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/sdlm.yaml  --exp_name SDLM --infer
```
