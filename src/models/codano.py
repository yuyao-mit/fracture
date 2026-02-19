
# https://arxiv.org/abs/2403.12553

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append("./neuraloperator")
from neuralop.models.codano import CODANO as original_CODANO


class CODANO(nn.Module):
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
            out_channels=C_in,
            kernel_size=1
        )

        self.codano = original_CODANO(
            output_variable_codimension=C_out,
            lifting_channels=64,
            hidden_variable_codimension=64,
            projection_channels=64,
            n_layers=6,
            n_modes=[[16, 16]] * 6,
            per_layer_scaling_factors=[[1.0, 1.0]] * 6,
            n_heads=[4] * 6,
            attention_scaling_factors=[1.0] * 6,
            nonlinear_attention=True,
            non_linearity=F.gelu,
            attention_token_dim=1,
            per_channel_attention=False,
            use_horizontal_skip_connection=True,
            horizontal_skips_map={5: 0, 4: 1, 3: 2},
            use_positional_encoding=True,
            positional_encoding_dim=2,
            positional_encoding_modes=None,
            static_channel_dim=0,
            domain_padding=0.1,
            domain_padding_mode="one-sided",
            layer_kwargs={},
            enable_cls_token=False
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T_in and C == self.C_in
        assert H == self.H and W == self.W

        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.temporal_fusion(x)
        y = self.codano(x)
        y = y.unsqueeze(1)
        return y
