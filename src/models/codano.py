

# https://arxiv.org/abs/2403.12553

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append("./neuraloperator")
from neuralop.models.codano import CODANO


class PhaseFieldPredictor(nn.Module):
    def __init__(self, CODANOClass=CODANO):
        super().__init__()

        self.input_steps  = 5
        self.output_steps = 1
        self.channels     = 10
        self.height       = 64
        self.width        = 64

        self.temporal_fusion = nn.Conv2d(
            in_channels=self.channels * self.input_steps,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )

        self.codano = CODANOClass(
            output_variable_codimension=1,
            lifting_channels=128,
            hidden_variable_codimension=32,
            projection_channels=128,
            n_layers=4,
            n_modes=[[8, 8], [8, 8], [8, 8], [8, 8]],
            per_layer_scaling_factors=[[1.0, 1.0]] * 4,
            n_heads=[2, 2, 2, 2],
            attention_scaling_factors=[1.0] * 4,
            nonlinear_attention=False,
            non_linearity=F.gelu,
            attention_token_dim=1,
            per_channel_attention=False,
            use_horizontal_skip_connection=True,
            horizontal_skips_map={3: 0, 2: 1},
            use_positional_encoding=False,
            positional_encoding_dim=8,
            positional_encoding_modes=None,
            static_channel_dim=0,
            domain_padding=0.1,
            domain_padding_mode="one-sided",
            layer_kwargs={},
            enable_cls_token=False
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.input_steps, f"Expected T={self.input_steps}, got {T}"
        assert C == self.channels,    f"Expected C={self.channels}, got {C}"
        assert H == self.height,      f"Expected H={self.height}, got {H}"
        assert W == self.width,       f"Expected W={self.width}, got {W}"

        x_flat = rearrange(x, 'b t c h w -> b (t c) h w')
        x_fused = self.temporal_fusion(x_flat)
        y = self.codano(x_fused)
        y = y.unsqueeze(1)
        return y
