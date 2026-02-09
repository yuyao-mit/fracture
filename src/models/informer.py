
# https://arxiv.org/abs/2012.07436

import torch
import torch.nn as nn
from einops import rearrange
from transformers import InformerConfig, InformerForPrediction

class FrameEncoder(nn.Module):
    def __init__(self, in_channels=10, embed_dim=256, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
    
    def forward(self, x):
        z = self.proj(x)
        tokens = rearrange(z, 'b d hp wp -> b (hp wp) d')
        hw = z.shape[-2:]
        return tokens, hw

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
        
        self.num_patches = (self.H // self.spatial_patch) * (self.W // self.spatial_patch)
        
        cfg = InformerConfig(
            input_size=self.embed_dim,
            context_length=self.T - 1,
            prediction_length=self.pred_len,
            lags_sequence=[1],
            d_model=256,
            encoder_layers=3,
            decoder_layers=3,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            attention_dropout=0.1,
            dropout=0.1,
            loss="nll",
            distribution_output="normal",
            num_time_features=1,
            num_dynamic_real_features=0,
        )
        self.informer = InformerForPrediction(cfg)
    
    @staticmethod
    def _build_time_features(bsz, past_len, future_len, device):
        total = past_len + future_len
        age = torch.linspace(0.0, 1.0, steps=total, device=device)
        age = age.unsqueeze(0).unsqueeze(-1).expand(bsz, total, 1)
        past_tf = age[:, :past_len, :]
        future_tf = age[:, past_len:, :]
        return past_tf, future_tf
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert (T, C, H, W) == (self.T, self.C, self.H, self.W)
        
        tokens_per_t, hw = [], None
        for t in range(T):
            tok, hw = self.enc(x[:, t])
            tokens_per_t.append(tok)
        
        zt = torch.stack(tokens_per_t, dim=1)
        
        seq = rearrange(zt, 'b t n d -> (b n) t d')
        BN, past_len, D = seq.shape
        device = seq.device
        
        past_tf, future_tf = self._build_time_features(BN, past_len, self.pred_len, device)
        
        past_mask = torch.ones(BN, past_len, D, device=device, dtype=torch.bool)
        
        out = self.informer(
            past_values=seq,
            past_time_features=past_tf,
            past_observed_mask=past_mask,
            future_time_features=future_tf
        )
        
        if hasattr(out, 'encoder_last_hidden_state') and out.encoder_last_hidden_state is not None:
            encoder_out = out.encoder_last_hidden_state
            if encoder_out.dim() == 3:
                last_hidden = encoder_out[:, -1, :]
                if last_hidden.shape[1] == D:
                    y_pred = last_hidden
                elif last_hidden.shape[1] > D:
                    y_pred = last_hidden[:, :D]
                else:
                    y_pred = torch.cat([
                        last_hidden, 
                        torch.zeros(BN, D - last_hidden.shape[1], device=device)
                    ], dim=1)
            else:
                raise ValueError(f"Unexpected encoder_last_hidden_state shape: {encoder_out.shape}")
                
        elif hasattr(out, 'loc') and out.loc is not None:
            loc = out.loc
            if loc.dim() == 1 and loc.shape[0] == BN:
                y_pred = loc.unsqueeze(-1).expand(-1, D)
            elif loc.dim() == 2:
                y_pred = loc if loc.shape[1] == D else loc[:, :D]
            else:
                raise ValueError(f"Cannot handle loc shape: {loc.shape}")
                
        else:
            raise ValueError("No suitable prediction tensor found in Informer output")
        
        y_tokens = rearrange(y_pred, '(b n) d -> b n d', b=B, n=self.num_patches)
        
        y_frame = self.dec(y_tokens, hw)
        
        return y_frame.unsqueeze(1)
    
    def generate_prediction(self, x):
        B, T, C, H, W = x.shape
        assert (T, C, H, W) == (self.T, self.C, self.H, self.W)
        
        tokens_per_t, hw = [], None
        for t in range(T):
            tok, hw = self.enc(x[:, t])
            tokens_per_t.append(tok)
        
        zt = torch.stack(tokens_per_t, dim=1)
        seq = rearrange(zt, 'b t n d -> (b n) t d')
        BN, past_len, D = seq.shape
        device = seq.device
        
        past_tf, future_tf = self._build_time_features(BN, past_len, self.pred_len, device)
        past_mask = torch.ones(BN, past_len, D, device=device, dtype=torch.bool)
        
        gen_out = self.informer.generate(
            past_values=seq,
            past_time_features=past_tf,
            past_observed_mask=past_mask,
            future_time_features=future_tf,
        )
        
        predictions = gen_out.sequences.mean(dim=1)
        y_pred = predictions[:, -1, :]
        
        y_tokens = rearrange(y_pred, '(b n) d -> b n d', b=B, n=self.num_patches)
        y_frame = self.dec(y_tokens, hw)
        
        return y_frame.unsqueeze(1)
