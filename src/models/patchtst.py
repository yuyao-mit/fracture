
# https://arxiv.org/abs/2211.14730

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PatchTSTConfig, PatchTSTForPrediction

class FrameEncoder(nn.Module):
    def __init__(self, in_channels=10, embed_dim=256, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        z = self.proj(x)
        tokens = rearrange(z, 'b d hp wp -> b (hp wp) d')
        hp, wp = z.shape[-2:]
        return tokens, (hp, wp)

class FrameDecoder(nn.Module):
    def __init__(self, out_channels=10, embed_dim=256, patch=4):
        super().__init__()
        self.deproj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch, stride=patch)
    def forward(self, tokens, hw):
        hp, wp = hw
        z = rearrange(tokens, 'b (hp wp) d -> b d hp wp', hp=hp, wp=wp)
        return self.deproj(z)

class PhaseFieldPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.C, self.T, self.H, self.W = 10, 5, 64, 64
        self.pred_len = 1
        self.embed_dim = 256
        self.spatial_patch = 4
        self.enc = FrameEncoder(self.C, self.embed_dim, self.spatial_patch)
        self.dec = FrameDecoder(self.C, self.embed_dim, self.spatial_patch)
        cfg = PatchTSTConfig(
            num_input_channels=self.embed_dim,
            context_length=self.T,
            prediction_length=self.pred_len,
            patch_length=1,
            patch_stride=1,
            d_model=256,
            num_hidden_layers=3,
            num_attention_heads=8,
            ffn_dim=512,
            norm_type="batchnorm",
            loss="mse",
        )
        self.tst = PatchTSTForPrediction(cfg)

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert (T, C, H, W) == (self.T, self.C, self.H, self.W)
        tokens_per_t, hw = [], None
        for t in range(T):
            tok, hw = self.enc(x[:, t])
            tokens_per_t.append(tok)
        zt = torch.stack(tokens_per_t, dim=1)
        seq = rearrange(zt, 'b t n d -> (b n) t d')
        out = self.tst(past_values=seq)
        y_tok = out.prediction_outputs
        y_next = y_tok[:, -1, :]
        y_tokens = rearrange(y_next, '(b n) d -> b n d', b=B)
        y_frame  = self.dec(y_tokens, hw)
        return y_frame.unsqueeze(1)
