# https://arxiv.org/abs/1801.07455

import torch
import torch.nn as nn
from itertools import product
from torch_geometric.nn import GCNConv

class GridGraph:
    def __init__(self, h_patch: int, w_patch: int):
        self.hp, self.wp = h_patch, w_patch
        V = h_patch * w_patch
        A = torch.zeros(2, V, V)
        A[0] = torch.eye(V)
        idx = lambda i, j: i * w_patch + j
        for i, j in product(range(h_patch), range(w_patch)):
            v = idx(i, j)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h_patch and 0 <= nj < w_patch:
                    A[1, v, idx(ni, nj)] = 1
        self.A = A

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_c, out_c, K, t_kernel=3):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.conv_t = nn.Conv2d(in_c, out_c * K, kernel_size=(t_kernel, 1), padding=(pad, 0))
        self.K = K

    def forward(self, x, A):
        x = x.contiguous()  # 保证连续
        n, c, t, v = x.shape
        x = self.conv_t(x).contiguous()
        x = x.view(n, self.K, -1, t, v).contiguous()
        A = A.to(dtype=x.dtype, device=x.device).contiguous()
        x = torch.einsum('nkctv,kvw->nctw', x, A).contiguous()
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, K, stride=1, residual=True):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_c, out_c, K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.25)
        )
        if not residual:
            self.res_conn = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.res_conn = lambda x: x
        else:
            self.res_conn = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, A):
        x = x.contiguous()
        y = self.gcn(x, A).contiguous()
        y = self.tcn(y).contiguous()
        res = self.res_conn(x.contiguous())
        if isinstance(res, torch.Tensor):
            res = res.contiguous()
        assert y.shape == res.shape, f"Shape mismatch: {y.shape} vs {res.shape}"
        y = y + res
        return self.act(y)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x.contiguous())

class PatchDecoder(nn.Module):
    def __init__(self, embed_dim, patch, out_ch):
        super().__init__()
        self.deproj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, z):
        return torch.sigmoid(self.deproj(z.contiguous()))

class STGCNVideoPredictor(nn.Module):
    def __init__(self, h, w, patch=8, embed_dim=64, channels_cfg=(64, 64, 128, 128, 256), t_in=5, in_ch=3):
        super().__init__()
        assert h % patch == 0 and w % patch == 0
        self.H, self.W, self.P, self.T, self.C = h, w, patch, t_in, in_ch
        self.hp, self.wp = h // patch, w // patch
        self.V = self.hp * self.wp
        A = GridGraph(self.hp, self.wp).A
        self.register_buffer('A', A)
        self.embed = PatchEmbed(in_ch, embed_dim, patch)
        self.decoder = PatchDecoder(channels_cfg[-1], patch, in_ch)
        chs = [embed_dim] + list(channels_cfg)
        self.blocks = nn.ModuleList([
            STGCNBlock(chs[i], chs[i+1], K=self.A.size(0))
            for i in range(len(chs) - 1)
        ])

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T and C == self.C and H == self.H and W == self.W
        x = x.reshape(-1, C, H, W).contiguous()
        x = self.embed(x).contiguous()
        Ce = x.size(1)
        x = x.view(B, self.T, Ce, self.V).permute(0, 2, 1, 3).contiguous()
        for blk in self.blocks:
            x = blk(x, self.A).contiguous()
        z = x[:, :, -1, :].contiguous().view(B, -1, self.hp, self.wp).contiguous()
        out = self.decoder(z).contiguous()
        return out.unsqueeze(1).contiguous()
