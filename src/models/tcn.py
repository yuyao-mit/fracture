
# https://arxiv.org/pdf/1803.01271

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding

        self.conv1 = weight_norm(
            nn.Conv3d(
                n_inputs,
                n_outputs,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                dilation=(dilation, 1, 1),
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout3d(dropout)

        self.conv2 = weight_norm(
            nn.Conv3d(
                n_outputs,
                n_outputs,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                dilation=(dilation, 1, 1),
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout3d(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv3d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def chomp(self, x):
        if self.padding == 0:
            return x
        return x[:, :, :-self.padding, :, :].contiguous()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(TCN, self).__init__()

        hidden_channels = 128
        num_levels = 3
        kernel_size = 2
        dropout = 0.2

        _, _, C_in, _, _ = input_shape
        _, _, C_out, _, _ = output_shape

        self.head = nn.Conv2d(C_in, hidden_channels, kernel_size=1)

        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = hidden_channels
            out_channels = hidden_channels
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.tail = nn.Conv2d(hidden_channels, C_out, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.head(x)
        x = x.view(B, T, -1, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.network(x)
        x = x[:, :, -1, :, :]
        x = self.tail(x)
        return x.unsqueeze(1)
