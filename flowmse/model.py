import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import numpy as np
import warnings
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model

class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--method", type=str, default="arfse", choices=["cfmse", "rfse", "arfse"], help="Framework: cfmse, rfse, or arfse")
        parser.add_argument("--path_type", type=str, default="denoising", choices=["generation", "denoising"], 
                            help="generation: source=z; denoising: source=y+z")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--t_eps", type=float, default=0.03)
        parser.add_argument("--T_rev", type=float, default=1.0)
        parser.add_argument("--num_eval_files", type=int, default=10)
        parser.add_argument("--loss_type", type=str, default="euclidean", choices=["mse", "mae", "euclidean"])
        parser.add_argument("--sigma", type=float, default=0.5, help="Sigma parameter for denoising path type")
        return parser

    def __init__(self, backbone, method="arfse", path_type="denoising", lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev=1.0, 
                 loss_type='euclidean', num_eval_files=10, sigma=0.5, data_module_cls=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['data_module_cls'])
        self.method = method
        self.path_type = path_type
        self.lr = lr
        self.sigma = sigma
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
      
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
    
        self.ema = ExponentialMovingAverage(self.parameters(), decay=ema_decay)
        self._error_loading_ema = False
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)


    def forward(self, x, t, y):
        t_input = None if self.method == "arfse" else t
        dnn_input = torch.cat([x, y], dim=1)
        return -self.dnn(dnn_input, t_input)

    def _step(self, batch, batch_idx):
        x0, y = batch
        z = torch.randn_like(x0)
        source = z if self.path_type == "generation" else y+self.sigma*z
        rdm = (1 - torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.T_rev, device=x0.device))
        t_view = t.view(-1, 1, 1, 1)

        if self.method == "cfmse":
            xt = (1 - t_view) * x0 + t_view * source
            condVF = (xt - x0) / t_view
        else:
            xt = (1 - t_view) * x0 + t_view * source
            condVF = source - x0
            # xt = t_view * x0 + (1 - t_view) * source
            # condVF = x0 - source
            
        vectorfield = self(xt, t, y)
        return self._loss(vectorfield, condVF)

    def _loss(self, vectorfield, condVF):
        if self.loss_type == 'mse':
            losses = torch.square((vectorfield - condVF).abs())
        elif self.loss_type == 'mae':
            losses = (vectorfield - condVF).abs()
        else: 
            err = vectorfield - condVF
            if torch.is_complex(err): err = torch.view_as_real(err)
            losses = torch.norm(err, p=2, dim=-1)
        return torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))


    def enhance(self, y, nfe=5):

        xt = torch.randn_like(y) if self.path_type == "generation" else y + torch.randn_like(y) * self.sigma

        dts = torch.full((nfe,), 1.0 / nfe, device=y.device)

        if self.method == "cfmse":
            curr_t = 1.0
            for step in range(nfe):
                dt = dts[step]
                t_tensor = torch.full((y.shape[0],), curr_t, device=y.device)
                v = self(xt, t_tensor, y)
                xt = xt - v * dt
                curr_t -= dt.item()
        
        elif self.method == "rfse":
            curr_t = 1.0
            for step in range(nfe):
                dt = dts[step]
                t_tensor = torch.full((y.shape[0],), curr_t, device=y.device)
                v = self(xt, t_tensor, y)
                xt = xt - v * dt
                curr_t -= dt.item()

        elif self.method == "arfse":
            for step in range(nfe):
                dt = dts[step]
                v = self(xt, None, y) 
                xt = xt - v * dt
                #xt = xt + v * dt
        
        return xt


    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_epoch=True)
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_epoch=True)
            self.log('si_sdr', si_sdr, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_load_checkpoint(self, checkpoint):
        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                self.ema.store(self.parameters())
                self.ema.copy_to(self.parameters())
            elif self.ema.collected_params is not None:
                self.ema.restore(self.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
