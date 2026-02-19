# https://arxiv.org/pdf/2211.14730

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PatchTSTConfig, PatchTSTForRegression


class PatchTST(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        patch_len=16,
        d_model=128,
        n_heads=4,
        num_layers=4,
    ):
        super().__init__()

        _, T, C, H, W = input_shape

        self.T = T
        self.C = C
        self.H = H
        self.W = W

        num_vars = C * H * W

        config = PatchTSTConfig(
            context_length=T,
            patch_len=patch_len,
            num_input_channels=num_vars,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=num_layers,
            num_targets=num_vars,
        )

        self.model = PatchTSTForRegression(config)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: [B, 1, C, H, W]
        """

        B, T, C, H, W = x.shape
        assert T == self.T

        # flatten spatial dims
        x_flat = rearrange(x, 'b t c h w -> b t (c h w)')

        outputs = self.model(past_values=x_flat)

        y_flat = outputs.regression_outputs   # [B, C*H*W]

        # reshape back to spatial field
        y = rearrange(y_flat, 'b (c h w) -> b 1 c h w',
                      c=self.C, h=self.H, w=self.W)

        return y
