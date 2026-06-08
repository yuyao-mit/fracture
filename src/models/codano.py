
# https://arxiv.org/abs/2403.12553

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURALOP = os.path.join(_HERE, "neuraloperator")
if _NEURALOP not in sys.path:
    sys.path.insert(0, _NEURALOP)

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from neuralop.models import CODANO as original_CODANO


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

        # CODANO treats each of the C_in fused channels as a "variable" token and
        # requires a stable id per variable (used to key its positional encodings).
        # Without this, use_positional_encoding=True iterates variable_ids=None and
        # raises `TypeError: 'NoneType' object is not iterable` at construction.
        self.var_ids = [f"v{i}" for i in range(C_in)]

        self.codano = original_CODANO(
            variable_ids=self.var_ids,
            output_variable_codimension=C_out,
            lifting_channels=64,
            # NOTE: at 256^2 this CODANO peaks at ~76 GB and OOMs on a 40 GB *and* 80 GB
            # A100 at batch 1-8. Reducing hidden_variable_codimension (64->16) does NOT
            # help — the footprint is codim-independent, so the driver is elsewhere
            # (spectral conv / codomain attention / domain_padding over the padded 282^2
            # grid). Making codano runnable needs a layer-by-layer memory profile, not a
            # width knob. Left at the original config pending that investigation.
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
            positional_encoding_modes=[16, 16],
            static_channel_dim=0,
            domain_padding=0.1,
            layer_kwargs={},
            enable_cls_token=False
        )

        # Recompute each CoDA layer in backward instead of storing its FFT/channel-MLP
        # activations — without this the 6-layer stack accumulates >>80 GB at 256^2 and
        # OOMs on any single GPU. Checkpointing cuts peak memory ~Nlayers-fold.
        self.codano.use_gradient_checkpointing = True

        # CODANO returns C_in * C_out channels (output_variable_codimension per input
        # variable); collapse the per-variable outputs to the C_out latent channels.
        self.output_proj = nn.Conv2d(C_in * C_out, C_out, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T_in and C == self.C_in
        assert H == self.H and W == self.W

        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.temporal_fusion(x)
        y = self.codano(x, input_variable_ids=self.var_ids)
        y = self.output_proj(y)
        y = y.unsqueeze(1)
        return y
