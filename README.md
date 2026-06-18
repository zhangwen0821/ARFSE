# ARFSE:Time-Unconditional Generative Speech Enhancement via Autonomous Rectified Flow

## 关键说明

ARFSE 是基于 `PyTorch Lightning` 的生成式语音增强项目，支持三种方法：`arfse`、`rfse`、`cfmse`。

## 依赖

```bash 
    pip install -r requirements.txt
``` 
## 目录

- `train.py`：训练入口
- `evaluate.py`：推理与评估
## 数据格式

```
<base_dir>/
  train/clean/*.wav
  train/noisy/*.wav
  valid/clean/*.wav
  valid/noisy/*.wav
  test/clean/*.wav
  test/noisy/*.wav
```

## 训练

```bash
python train.py --base_dir /path/to/dataset --backbone ncsnpp
```

常用参数：
- `--backbone`：ncsnpp / ncsnpplarge / ncsnpp12M / ncsnpp6M / dcunet
- `--method`：arfse / rfse / cfmse
- `--path_type`：denoising / generation
- `--lr`：1e-4
- `--batch_size`：4
- `--n_fft`：510
- `--hop_length`：128
- `--num_frames`：256

## 推理

```bash
python evaluate.py --test_dir /path/to/dataset --ckpt /path/to/checkpoint.ckpt --folder_destination /path/to/output --N 5
```

## 模型权重

Hugging Face: https://huggingface.co/zhangwen0821/ARFSE

## Demo

Github-pages: https://zhangwen0821.github.io/ARFSE/

## 参考链接

[dns2020testset](https://github.com/microsoft/DNS-Challenge.git)

[flowmse](https://github.com/seongq/flowmse.git)

[sgmse-bbed](https://github.com/sp-uhh/sgmse-bbed.git)

[sgmse](https://github.com/sp-uhh/sgmse.git)
