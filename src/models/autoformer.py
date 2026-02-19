# https://arxiv.org/abs/2106.13008

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoformerConfig, AutoformerForPrediction


class Autoformer(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        patch=8,
        d_model=128,
        n_heads=4,
        num_layers=3,
    ):
        super().__init__()

        _, T, C, H, W = input_shape

        assert H % patch == 0 and W % patch == 0

        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.patch = patch

        self.hp = H // patch
        self.wp = W // patch
        self.num_vars = C * self.hp * self.wp

        config = AutoformerConfig(
            context_length=T,
            prediction_length=1,
            input_size=self.num_vars,
            d_model=d_model,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            encoder_attention_heads=n_heads,
            decoder_attention_heads=n_heads,
            feature_size=0,
        )

        self.model = AutoformerForPrediction(config)

        # patch embedding
        self.patch_embed = nn.Conv2d(C, C, kernel_size=patch, stride=patch)

        # patch decoder
        self.patch_decode = nn.ConvTranspose2d(
            C, C, kernel_size=patch, stride=patch
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: [B, 1, C, H, W]
        """

        B, T, C, H, W = x.shape

        # ---- Patchify ----
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = rearrange(
            x,
            '(b t) c hp wp -> b t (c hp wp)',
            b=B, t=T
        )
        past_time_features = torch.zeros(
            B, T, 0, device=x.device
        )
        past_observed_mask = torch.ones(
            B, T, self.num_vars,
            device=x.device
        )

        # ---- Autoformer ----
        outputs = self.model(
            past_values=x,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        )

        # mean prediction
        prediction = outputs.sequences.mean(dim=1)

        # ---- reshape back ----
        prediction = rearrange(
            prediction,
            'b (c hp wp) -> b c hp wp',
            c=self.C, hp=self.hp, wp=self.wp
        )

        prediction = self.patch_decode(prediction)

        return prediction.unsqueeze(1)
