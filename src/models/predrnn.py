

# https://arxiv.org/abs/2103.09504

import torch
import torch.nn as nn

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
        
class PredRNN(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        num_layers=3,
        num_hidden=[64, 64, 64],
        filter_size=5,
        stride=1,
        layer_norm=True,
    ):
        super(PredRNN, self).__init__()

        _, T, C, H, W = input_shape

        self.input_steps = T
        self.output_steps = output_shape[1]
        self.frame_channel = C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.height = H
        self.width = W

        cell_list = []

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel=in_channel,
                    num_hidden=num_hidden[i],
                    width=W,
                    filter_size=filter_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(
            num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )


    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: [B, 1, C, H, W]
        """

        B, T, C, H, W = x.shape
        frames = x.permute(0, 1, 3, 4, 2).contiguous()
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros(
                B, self.num_hidden[i], H, W, device=x.device
            )
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            B, self.num_hidden[0], H, W, device=x.device
        )

        for t in range(T):
            net = frames[:, t]

            h_t[0], c_t[0], memory = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

        # ---- one-step prediction ----
        delta = self.conv_last(h_t[-1])  # [B, frame_channel, H, W]
        delta = delta.permute(0, 2, 3, 1).contiguous()
        phi_last = frames[:, -1]
        phi_next = phi_last + delta  
        phi_next = phi_next.permute(0, 3, 1, 2).unsqueeze(1)

        return phi_next
