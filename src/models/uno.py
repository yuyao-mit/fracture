
# https://arxiv.org/abs/2204.11127

import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("./neuraloperator")
from neuralop.models.uno import UNO
from einops import rearrange

class PhaseFieldPredictor(nn.Module):
    def __init__(self, UNOClass=UNO):
        super().__init__()
        
        self.input_steps = 5
        self.output_steps = 1
        self.channels = 10
        self.height = 64
        self.width = 64
        
        self.temporal_fusion = nn.Conv2d(
            in_channels=self.channels * self.input_steps,
            out_channels=self.channels * 2,
            kernel_size=3,
            padding=1
        )
        
        self.uno = UNOClass(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            hidden_channels=64,
            lifting_channels=128,
            projection_channels=128,
            n_layers=4,
            uno_out_channels=[64, 64, 64, 64],
            uno_n_modes=[[8, 8], [8, 8], [8, 8], [8, 8]],
            uno_scalings=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            horizontal_skips_map={3: 0, 2: 1},
            domain_padding=0.1,
            positional_embedding='grid',
            non_linearity=F.gelu,
            channel_mlp_skip="linear"
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.input_steps, f"Expected T={self.input_steps}, got {T}"
        assert C == self.channels, f"Expected C={self.channels}, got {C}"
        assert H == self.height, f"Expected H={self.height}, got {H}"
        assert W == self.width, f"Expected W={self.width}, got {W}"
        
        x_flattened = rearrange(x, 'b t c h w -> b (t c) h w')
        x_fused = self.temporal_fusion(x_flattened)
        output = self.uno(x_fused)
        output = output.unsqueeze(1)
        
        return output
