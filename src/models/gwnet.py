
# https://arxiv.org/abs/1906.00121

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class GWNet(nn.Module):
    def __init__(self, input_shape, output_shape, device):
        super(GWNet, self).__init__()

        B, T_in, C_in, H, W = input_shape
        _, T_out, C_out, _, _ = output_shape

        assert T_out == 1

        self.T_in = T_in
        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.N = H * W

        self.dropout = 0.3
        self.blocks = 4
        self.layers = 2
        self.kernel_size = 2
        self.residual_channels = 32
        self.dilation_channels = 32
        self.skip_channels = 256
        self.end_channels = 512
        self.gcn_bool = True
        self.addaptadj = True

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(C_in, self.residual_channels, kernel_size=(1,1))

        self.nodevec1 = nn.Parameter(torch.randn(self.N, 10).to(device), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.N).to(device), requires_grad=True)

        receptive_field = 1
        self.supports_len = 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                self.filter_convs.append(
                    nn.Conv2d(self.residual_channels,
                              self.dilation_channels,
                              kernel_size=(1,self.kernel_size),
                              dilation=new_dilation)
                )

                self.gate_convs.append(
                    nn.Conv2d(self.residual_channels,
                              self.dilation_channels,
                              kernel_size=(1,self.kernel_size),
                              dilation=new_dilation)
                )

                self.residual_convs.append(
                    nn.Conv2d(self.dilation_channels,
                              self.residual_channels,
                              kernel_size=(1,1))
                )

                self.skip_convs.append(
                    nn.Conv2d(self.dilation_channels,
                              self.skip_channels,
                              kernel_size=(1,1))
                )

                self.bn.append(nn.BatchNorm2d(self.residual_channels))

                self.gconv.append(
                    gcn(self.dilation_channels,
                        self.residual_channels,
                        self.dropout,
                        support_len=self.supports_len,
                        order=2)
                )

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(self.skip_channels,
                                    self.end_channels,
                                    kernel_size=(1,1))

        self.end_conv_2 = nn.Conv2d(self.end_channels,
                                    C_out,
                                    kernel_size=(1,1))

        self.receptive_field = receptive_field

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T_in and C == self.C_in
        assert H == self.H and W == self.W

        x = x.permute(0,2,3,4,1).reshape(B, C, self.N, T)

        if T < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T, 0, 0, 0))

        x = self.start_conv(x)
        skip = 0

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            s = self.skip_convs[i](x)
            if isinstance(skip, int):
                skip = s
            else:
                skip = skip[:, :, :, -s.size(3):] + s

            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        x = x[:, :, :, -1]
        x = x.reshape(B, self.C_out, self.H, self.W).unsqueeze(1)

        return x
