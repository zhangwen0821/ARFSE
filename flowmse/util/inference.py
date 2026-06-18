import torch
from torchaudio import load
from pesq import pesq
from pystoi import stoi
from .other import si_sdr, pad_spec

sr = 16000

def evaluate_model(model, num_eval_files, nfe=5):
    
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    indices = torch.linspace(0, len(clean_files) - 1, num_eval_files, dtype=torch.long)
    selected_clean = [clean_files[i] for i in indices]
    selected_noisy = [noisy_files[i] for i in indices]

    _pesq, _si_sdr, _estoi = 0.0, 0.0, 0.0
    model.eval()
  
    for clean_file, noisy_file in zip(selected_clean, selected_noisy):
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        T_orig = x.size(1)   

        norm_factor = y.abs().max() + 1e-8
        y_norm = y / norm_factor

        Y = model._forward_transform(model._stft(y_norm.cuda()))
        Y = pad_spec(Y.unsqueeze(0)) # [1, C, F, T]

        with torch.no_grad():
            sample = model.enhance(Y, nfe=nfe) 

        x_hat = model.to_audio(sample.squeeze(0), T_orig) 
        x_hat = (x_hat * norm_factor).squeeze().cpu().numpy()
        
        x = x.squeeze().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)

    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files