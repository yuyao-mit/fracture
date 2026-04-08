
# https://arxiv.org/abs/2010.08895

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURALOP = os.path.join(_HERE, "neuraloperator")
if _NEURALOP not in sys.path:
    sys.path.insert(0, _NEURALOP)

import torch
from torch import nn
import torch.nn.functional as F
from neuralop.models import FNO as original_FNO

class FNO(nn.Module):
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

        self.model = original_FNO(
            n_modes=(12, 12),
            hidden_channels=32,
            in_channels=T_in * C_in,
            out_channels=C_out,
            n_layers=4
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        x = self.model(x)
        x = x.unsqueeze(1)
        return x
