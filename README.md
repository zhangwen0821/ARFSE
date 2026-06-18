# ARFSE: Time-Unconditional Generative Speech Enhancement via Autonomous Rectified Flow

ARFSE is a generative speech enhancement project based on `PyTorch Lightning`, supporting three methods: `arfse`, `rfse`, and `cfmse`.

## Demo

GitHub Pages: https://zhangwen0821.github.io/ARFSE/

## Dependencies

```bash
pip install -r requirements.txt
```

## Data Format

```
<base_dir>/
  train/clean/*.wav
  train/noisy/*.wav
  valid/clean/*.wav
  valid/noisy/*.wav
  test/clean/*.wav
  test/noisy/*.wav
```

## Training

```bash
python train.py --base_dir /path/to/dataset --backbone ncsnpp
```

Common options:
- `--backbone`: ncsnpp / ncsnpplarge / ncsnpp12M / ncsnpp6M / dcunet
- `--method`: arfse / rfse / cfmse
- `--path_type`: denoising / generation
- `--lr`: 1e-4
- `--batch_size`: 4

## Inference

```bash
python evaluate.py --test_dir /path/to/dataset --ckpt /path/to/checkpoint.ckpt --folder_destination /path/to/output --N 5
```

## Model Weights

Hugging Face: https://huggingface.co/zhangwen0821/ARFSE

## References

- [dns2020testset](https://github.com/microsoft/DNS-Challenge.git)
- [flowmse](https://github.com/seongq/flowmse.git)
- [sgmse-bbed](https://github.com/sp-uhh/sgmse-bbed.git)
- [sgmse](https://github.com/sp-uhh/sgmse.git)
