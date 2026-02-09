

# https://arxiv.org/abs/1906.00121

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, List, Tuple
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        assert x.dim() == 4, f"x must be [N,C,V,L], got {tuple(x.shape)}"
        assert A.dim() == 2, f"A must be [V,V], got {tuple(A.shape)}"
        N, C, V, L = x.shape
        assert A.shape[0] == V, f"A[0]={A.shape[0]} must match V={V}"
        if A.device != x.device or A.dtype != x.dtype:
            A = A.to(device=x.device, dtype=x.dtype)
        x = x.contiguous()
        A = A.contiguous()
        y = torch.einsum('ncvl,vw->ncwl', (x, A))
        return y.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        assert isinstance(support, (list, tuple)) and len(support) > 0, \
            f"support must be non-empty list/tuple, got {type(support)}"
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1).contiguous()
        h = self.mlp(h.contiguous())
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, 
                 aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, 
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.dtype = torch.float32

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        self.supports = None
        self.supports_len = 0
        if supports is not None:
            buf_list = []
            for i, A in enumerate(supports):
                A = torch.as_tensor(A).to(device=self.device, dtype=self.dtype)
                self.register_buffer(f"_support_{i}", A.contiguous())
                buf_list.append(getattr(self, f"_support_{i}"))
            self.supports = buf_list
            self.supports_len += len(buf_list)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10, device=self.device, dtype=self.dtype), requires_grad=True
                )
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes, device=self.device, dtype=self.dtype), requires_grad=True
                )
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                A0 = torch.as_tensor(aptinit, dtype=self.dtype, device=self.device)
                U, S, Vh = torch.linalg.svd(A0, full_matrices=False)
                initemb1 = U[:, :10] @ torch.diag(S[:10].pow(0.5))
                initemb2 = torch.diag(S[:10].pow(0.5)) @ Vh[:10, :]
                self.nodevec1 = nn.Parameter(initemb1.contiguous(), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.contiguous(), requires_grad=True)
                self.supports_len += 1
        receptive_field = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=new_dilation
                ))
                self.gate_convs.append(nn.Conv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=new_dilation
                ))
                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, 1)
                ))
                self.skip_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=(1, 1)
                ))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(
                        dilation_channels, residual_channels, dropout, support_len=self.supports_len
                    ))

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True
        )

        self.receptive_field = receptive_field

    def _gather_runtime_supports(self, x):
        runtime_supports = None
        if self.gcn_bool:
            if self.addaptadj:
                base = self.supports if self.supports is not None else []
                adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
                adp = adp.to(device=x.device, dtype=x.dtype).contiguous()
                runtime_supports = list(base) + [adp]
            else:
                if self.supports is not None:
                    aligned = []
                    for A in self.supports:
                        if A.device != x.device or A.dtype != x.dtype:
                            aligned.append(A.to(device=x.device, dtype=x.dtype))
                        else:
                            aligned.append(A)
                    runtime_supports = aligned
        return runtime_supports

    def forward(self, input):
        x = input
        assert x.dim() == 4, f"Expect input [N,C,V,T], got {tuple(x.shape)}"
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        in_len = x.size(3)
        assert self.receptive_field >= 1, "receptive_field must be >= 1"
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x.contiguous())

        skip = None
        runtime_supports = self._gather_runtime_supports(x)
        autocast_disable = torch.cuda.is_available()
        autocast_ctx = torch.cuda.amp.autocast(enabled=False) if autocast_disable else torch.autocast("cpu", enabled=False)
        with autocast_ctx:
            for i in range(self.blocks * self.layers):
                residual = x.contiguous()
                f = self.filter_convs[i](residual.contiguous())
                g = self.gate_convs[i](residual.contiguous())
                x = (torch.tanh(f) * torch.sigmoid(g)).contiguous()
                s = self.skip_convs[i](x.contiguous())
                if isinstance(skip, torch.Tensor) and skip.size(3) >= s.size(3):
                    skip = skip[:, :, :, -s.size(3):].contiguous()
                else:
                    skip = 0
                skip = (s + skip) if isinstance(skip, torch.Tensor) else s

                if self.gcn_bool and (runtime_supports is not None) and len(runtime_supports) > 0:
                    x = self.gconv[i](x.contiguous(), runtime_supports).contiguous()
                else:
                    x = self.residual_convs[i](x.contiguous()).contiguous()
                assert x.size(3) > 0, "Time dimension collapsed to 0; check kernel/dilation/receptive_field."
                x = (x + residual[:, :, :, -x.size(3):]).contiguous()
                x = self.bn[i](x.contiguous())
            x = F.relu(skip.contiguous())
            x = F.relu(self.end_conv_1(x.contiguous()))
            x = self.end_conv_2(x.contiguous())
        return x


class ConvProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 spatial_size: Tuple[int, int], target_size: Tuple[int, int]):
        super(ConvProjector, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.target_size = target_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.encoder(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.decoder(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x

class PhaseFieldPredictor(nn.Module):
    def __init__(self,
                 input_channels: int = 10,
                 output_channels: int = 10,
                 spatial_size: Tuple[int, int] = (64, 64),
                 proj_channels: int = 32,
                 target_spatial_size: Tuple[int, int] = (32, 32),
                 input_seq_len: int = 5,
                 output_seq_len: int = 1,
                 dropout: float = 0.3,
                 gcn_bool: bool = True,
                 addaptadj: bool = True,
                 residual_channels: int = 32,
                 dilation_channels: int = 32,
                 skip_channels: int = 256,
                 end_channels: int = 512,
                 kernel_size: int = 2,
                 blocks: int = 4,
                 layers: int = 2,
                 device: str = 'cpu'):
        super(PhaseFieldPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.proj_channels = proj_channels
        self.target_spatial_size = target_spatial_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.device = device
        
        self.projector = ConvProjector(
            in_channels=input_channels,
            out_channels=proj_channels,
            spatial_size=spatial_size,
            target_size=target_spatial_size
        )
        
        num_nodes = target_spatial_size[0] * target_spatial_size[1]
        
        self.gwnet = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=None,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=None,
            in_dim=proj_channels,
            out_dim=proj_channels * output_seq_len,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_proj = self.projector.encode(x)
        x_gwnet = rearrange(x_proj, 'b t c h w -> b c (h w) t')
        pred_gwnet = self.gwnet(x_gwnet)
        pred_gwnet = rearrange(pred_gwnet, 'b (t c) n t_out -> b t c n t_out', 
                              t=self.output_seq_len, c=self.proj_channels)
        H_small, W_small = self.target_spatial_size
        pred_gwnet = pred_gwnet[:, :, :, :, -1]
        pred_proj = rearrange(pred_gwnet, 'b t c (h w) -> b t c h w', h=H_small, w=W_small)
        output = self.projector.decode(pred_proj)
        return output
    
    def get_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': f'({self.input_seq_len}, {self.input_channels}, {self.spatial_size[0]}, {self.spatial_size[1]})',
            'output_shape': f'({self.output_seq_len}, {self.output_channels}, {self.spatial_size[0]}, {self.spatial_size[1]})',
            'compressed_spatial_size': self.target_spatial_size,
            'projection_channels': self.proj_channels
        }
