
# https://arxiv.org/abs/2308.08794

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./neuraloperator")
from neuralop.models import RNO as original_RNO


class RNO(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        B, T_in, C_in, H, W = input_shape
        B2, T_out, C_out, H2, W2 = output_shape

        assert H == H2 and W == W2
        assert B == B2

        self.T_in = T_in
        self.T_out = T_out
        self.C_in = C_in
        self.C_out = C_out
        n_modes = (16, 16)
        hidden_channels = 32               # KS setting in paper
        n_layers = 4                       # default in paper

        self.model = original_RNO(
            n_modes=n_modes,
            in_channels=C_in,
            out_channels=C_out,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            rno_skip=False,

            lifting_channel_ratio=2,
            projection_channel_ratio=2,

            positional_embedding="grid",

            non_linearity=F.selu,
            norm=None,

            complex_data=False,

            use_channel_mlp=True,
            channel_mlp_dropout=0.0,
            channel_mlp_expansion=0.5,

            channel_mlp_skip="soft-gating",
            fno_skip="linear",

            return_sequences=False,

            resolution_scaling_factor=None,
            domain_padding=None,

            fno_block_precision="full",
            stabilizer=None,

            factorization=None,
            rank=1.0,
            separable=False,
            preactivation=False,
        )

    def forward(self, x):
        B = x.shape[0]

        outputs = []
        states = None

        for t in range(self.T_out):
            pred, states = self.model(
                x,
                init_hidden_states=states,
                return_hidden_states=True,
                keep_states_padded=True,
            )

            outputs.append(pred)
            x = pred.unsqueeze(1)

        return torch.stack(outputs, dim=1)
