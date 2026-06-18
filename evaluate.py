import time
import glob
import torch
import os
import re
import pandas as pd
from argparse import ArgumentParser
from os.path import join
from soundfile import write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
from flowmse.model import VFModel
from flowmse.util.other import pad_spec
from utils import energy_ratios, ensure_dir, print_mean_std
from flowmse.data_module import SpecsDataModule
torch.serialization.add_safe_globals([SpecsDataModule])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Test data directory')
    parser.add_argument("--ckpt", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--folder_destination", type=str, required=True, help="Output destination")
    parser.add_argument("--N", type=int, default=5, help="Number of inference steps (NFE)")
    args = parser.parse_args()

    sr = 16000
    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    os.makedirs(join(args.folder_destination, "files"), exist_ok=True)
    
    model = VFModel.load_from_checkpoint(
        args.ckpt, 
        base_dir="",
        data_module_cls=SpecsDataModule,
        strict=False
    )

    model.eval()
    model.cuda()

    match = re.search(r'epoch=(\d+)', args.ckpt)
    epoch = match.group(1) if match else "unknown"

    noisy_files = sorted(glob.glob(f'{noisy_dir}/*.wav'))
    
    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    total_audio_sec, total_inference_sec, n_utt_rtf = 0.0, 0.0, 0
    
    print(f"Running Inference with Method: {model.method.upper()}, NFE: {args.N}")

    with torch.no_grad():
        for cnt, noisy_file in tqdm(enumerate(noisy_files)):
            filename = os.path.basename(noisy_file)
            
            x, _ = load(join(clean_dir, filename))
            y, _ = load(noisy_file)
            T_orig = y.size(1)

            current_audio_duration = T_orig / sr
            torch.cuda.synchronize()
            start_time = time.time()

            norm_factor = y.abs().max().item() + 1e-8
            y_in = y / norm_factor
            Y = model._forward_transform(model._stft(y_in.cuda()))
            Y = pad_spec(Y.unsqueeze(0))
            
            sample = model.enhance(Y, nfe=args.N)
            
            x_hat_spec = sample.squeeze(0)
            x_hat = model.to_audio(x_hat_spec, T_orig)
            
            torch.cuda.synchronize()
            end_time = time.time()


            if cnt > 0:
                total_audio_sec += current_audio_duration
                total_inference_sec += (end_time - start_time)
                n_utt_rtf += 1

            x_hat_np = (x_hat * norm_factor).squeeze().cpu().numpy()
            x_np = x.squeeze().numpy()
            y_np = y.squeeze().numpy()
            noise_np = y_np - x_np

            write(join(args.folder_destination, "files", filename), x_hat_np, sr)

            data["filename"].append(filename)
            try:
                data["pesq"].append(pesq(sr, x_np, x_hat_np, 'wb'))
            except:
                data["pesq"].append(float("nan"))
            
            data["estoi"].append(stoi(x_np, x_hat_np, sr, extended=True))
            
            try:
                sdr, sir, sar = energy_ratios(x_hat_np, x_np, noise_np)
                data["si_sdr"].append(sdr)
                data["si_sir"].append(sir)
                data["si_sar"].append(sar)
            except:
                for k in ["si_sdr", "si_sir", "si_sar"]: data[k].append(float("nan"))

    rtf = total_inference_sec / total_audio_sec if total_audio_sec > 0 else 0


    df = pd.DataFrame(data)
    df.to_csv(join(args.folder_destination, "_results.csv"), index=False)


    with open(join(args.folder_destination, "_avg_results.txt"), 'w') as f:
        for metric in ["pesq", "estoi", "si_sdr", "si_sir", "si_sar"]:
            f.write(f"{metric.upper()}: {print_mean_std(data[metric])}\n")
        f.write(f"RTF: {rtf:.4f}\n")

    with open(join(args.folder_destination, "_settings.txt"), 'w') as f:
        f.write(f"Method: {model.method}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"NFE: {args.N}\n")
        f.write(f"RTF: {rtf:.4f}\n")

    print(f"\nInference Complete | Method: {model.method} | RTF: {rtf:.4f}")