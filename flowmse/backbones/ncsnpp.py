# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modified for dynamic discriminative/generative switching.

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
#from sgmse.util.other import pad_spec

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model - Supports dynamic discriminative/generative mode via time_cond."""

    def __init__(self, 
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        input_channels = 4,
        spatial_channels = 1,
        dropout = .0,
        centered = False,
        **kwargs):
        super().__init__()

        self.FORCE_STFT_OUT = False
        self.act = act = get_act(nonlinearity)

        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = [image_size // (2 ** i) for i in range(self.num_resolutions)]
        
        self.conditional = conditional  
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        self.resblock_type = resblock_type.lower()
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type.lower()
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.total_channels = self.input_channels * self.spatial_channels

        self.output_layer = nn.Conv2d(self.total_channels, 2*self.spatial_channels, 1)

        modules = []

        # --- Module Definitions ---
        AttnBlock = functools.partial(layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)
        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        
        if self.progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        
        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        
        if self.progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)

        if self.resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, dropout=dropout, init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
        else:
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel, init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)

        # --- Embedding Modules ---
        if self.embedding_type == 'fourier':
            modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf
        elif self.embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        # --- Downsampling Path ---
        modules.append(conv3x3(self.total_channels, nf))
        hs_c = [nf]
        in_ch = nf
        combiner = functools.partial(Combine, method=progressive_combine.lower())

        for i_level in range(self.num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if self.progressive_input == 'input_skip':
                    modules.append(combiner(dim1=self.total_channels, dim2=in_ch))
                    if progressive_combine.lower() == 'cat': in_ch *= 2
                elif self.progressive_input == 'residual':
                    # Simplified for clarity, original logic preserved via list append
                    modules.append(functools.partial(layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)(in_ch=self.total_channels, out_ch=in_ch))
                hs_c.append(in_ch)

        # --- Middle Blocks ---
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # --- Upsampling Path ---
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=nf * ch_mult[i_level]))
                in_ch = nf * ch_mult[i_level]
            
            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if self.progressive != 'none':
                modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                modules.append(conv3x3(in_ch, self.total_channels if self.progressive=='output_skip' else in_ch, init_scale=init_scale))
                if self.progressive == 'residual' and i_level != 0:
                    modules.append(layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=True))

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        if self.progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond=None):
        modules = self.all_modules
        m_idx = 0

        # Consistent channel conversion (4-channel logic)
        x_chans = []
        for chan in range(self.spatial_channels):
            x_chans.append(torch.cat([ 
                torch.cat([x[:,[chan+in_chan],:,:].real, x[:,[chan+in_chan],:,:].imag ], dim=1) 
                for in_chan in range(self.input_channels // 2)], dim=1)
            )
        x = torch.cat(x_chans, dim=1) 

        # --- Dynamic Timestep Embedding ---
        temb = None
        if time_cond is not None:
            if self.embedding_type == 'fourier':
                temb = modules[m_idx](torch.log(time_cond))
            elif self.embedding_type == 'positional':
                # Note: self.sigmas assumed to exist if using positional + generative
                temb = layers.get_timestep_embedding(time_cond, self.nf)
            m_idx += 1
            if self.conditional:
                temb = modules[m_idx](temb) ; m_idx += 1
                temb = modules[m_idx](self.act(temb)) ; m_idx += 1
        else:
            # Skip indices to keep module alignment for the rest of the network
            m_idx += 1 
            if self.conditional: m_idx += 2

        if not self.centered:
            x = 2 * x - 1.

        input_pyramid = x if self.progressive_input != 'none' else None
        hs = [modules[m_idx](x)] ; m_idx += 1

        # --- Downsampling ---
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb) ; m_idx += 1
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h) ; m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                else:
                    h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                
                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h) ; m_idx += 1
                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid) ; m_idx += 1
                    h = (input_pyramid + h) / np.sqrt(2.) if self.skip_rescale else input_pyramid + h
                    input_pyramid = h
                hs.append(h)

        # --- Middle ---
        h = hs[-1]
        h = modules[m_idx](h, temb) ; m_idx += 1
        h = modules[m_idx](h) ; m_idx += 1
        h = modules[m_idx](h, temb) ; m_idx += 1

        # --- Upsampling ---
        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb) ; m_idx += 1
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h) ; m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    pyramid = self.act(modules[m_idx](h)) ; m_idx += 1
                    pyramid = modules[m_idx](pyramid) ; m_idx += 1
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h)) ; m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h) ; m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid) ; m_idx += 1
                        pyramid = (pyramid + h) / np.sqrt(2.) if self.skip_rescale else pyramid + h
                        h = pyramid

            if i_level != 0:
                h = modules[m_idx](h) if self.resblock_type == 'ddpm' else modules[m_idx](h, temb)
                m_idx += 1

        # --- Final Output ---
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h)) ; m_idx += 1
            h = modules[m_idx](h) ; m_idx += 1

        if self.scale_by_sigma and time_cond is not None:
            h = h / time_cond.reshape((x.shape[0], *([1] * len(x.shape[1:]))))

        h = self.output_layer(h)
        h = torch.reshape(h, (h.size(0), 2, self.spatial_channels, h.size(2), h.size(3)))
        h = torch.permute(h, (0, 2, 3, 4, 1)).contiguous()
        return torch.view_as_complex(h)

    @staticmethod
    def add_argparse_args(parser):
        return parser

@BackboneRegistry.register("ncsnpplarge")
class NCSNppLarge(NCSNpp):
    def __init__(self, **kwargs):
        super().__init__(nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,), **kwargs)
    
    @staticmethod
    def add_argparse_args(parser):
        return parser

@BackboneRegistry.register("ncsnpp12M")
class NCSNpp12M(NCSNpp):
    def __init__(self, **kwargs):
        super().__init__(nf=96, ch_mult=(1, 2, 2, 1), num_res_blocks=1, attn_resolutions=(0,), **kwargs)
    
    @staticmethod
    def add_argparse_args(parser):
        return parser

@BackboneRegistry.register("ncsnpp6M")
class NCSNpp6M(NCSNpp):
    def __init__(self, **kwargs):
        super().__init__(nf=96, ch_mult=(1, 1, 1, 1), num_res_blocks=1, attn_resolutions=(0,), **kwargs)
    
    @staticmethod
    def add_argparse_args(parser):
        return parser