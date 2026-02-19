
# https://arxiv.org/abs/2204.11127

import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("./neuraloperator")

from neuralop.models.uno import UNO as original_UNO
from einops import rearrange


class UNO(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        _, T_in, C_in, H, W = input_shape
        _, T_out, C_out, H_out, W_out = output_shape

        assert H == H_out and W == W_out
        assert T_out == 1

        self.T_in = T_in
        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W

        self.temporal_fusion = nn.Conv2d(
            in_channels=T_in * C_in,
            out_channels=32,
            kernel_size=1
        )

        self.uno = original_UNO(
            in_channels=32,
            out_channels=C_out,
            hidden_channels=64,
            lifting_channels=32,
            projection_channels=32,
            n_layers=7,
            uno_out_channels=[64, 128, 128, 128, 128, 64, 32],
            uno_n_modes=[[12, 12]] * 7,
            uno_scalings=[
                [0.75, 0.75],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [4.0/3.0, 4.0/3.0],
            ],
            horizontal_skips_map={4: 2, 5: 1, 6: 0},
            domain_padding=0.1,
            positional_embedding='grid',
            non_linearity=F.gelu,
            channel_mlp_skip="linear"
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T_in and C == self.C_in
        assert H == self.H and W == self.W

        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.temporal_fusion(x)
        out = self.uno(x)
        out = out.unsqueeze(1)
        return out
