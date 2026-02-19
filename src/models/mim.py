
# https://arxiv.org/abs/1811.07490

import torch
import torch.nn as nn
import math

class ConvLN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_norm):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=padding
        )
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W]
        if self.layer_norm:
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
        return x

class MIMBlock(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, tln=False):
        super().__init__()

        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self.layer_norm = tln
        self._forget_bias = 1.0

        self.h_conv = ConvLN(num_hidden, num_hidden * 4, filter_size, tln)
        self.x_conv = ConvLN(num_hidden, num_hidden * 4, filter_size, tln)

        self.t_conv = ConvLN(num_hidden, num_hidden * 3, filter_size, tln)
        self.s_conv = ConvLN(num_hidden, num_hidden * 4, filter_size, tln)
        self.x2_conv = ConvLN(num_hidden_in, num_hidden * 4, filter_size, tln)

        self.cell_reduce = nn.Conv2d(num_hidden * 2, num_hidden, 1)

        self.ct_weight = nn.Parameter(
            torch.randn(1, num_hidden * 2, self.height, self.width)
        )
        self.oc_weight = nn.Parameter(
            torch.randn(1, num_hidden, self.height, self.width)
        )

        self.convlstm_c = None

    def init_state(self, device, batch):
        return torch.zeros(
            batch, self.num_hidden, self.height, self.width,
            device=device
        )

    def MIMS(self, x, h_t, c_t):
        B = x.size(0)

        if h_t is None:
            h_t = self.init_state(x.device, B)
        if c_t is None:
            c_t = self.init_state(x.device, B)

        h_concat = self.h_conv(h_t)
        i_h, g_h, f_h, o_h = torch.chunk(h_concat, 4, dim=1)

        ct_activation = torch.cat([c_t, c_t], dim=1) * self.ct_weight
        i_c, f_c = torch.chunk(ct_activation, 2, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.x_conv(x)
            i_x, g_x, f_x, o_x = torch.chunk(x_concat, 4, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = c_new * self.oc_weight
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        device = x.device
        B = x.size(0)

        if h is None:
            h = self.init_state(device, B)
        if c is None:
            c = self.init_state(device, B)
        if m is None:
            m = self.init_state(device, B)
        if diff_h is None:
            diff_h = torch.zeros_like(h)

        t_cc = self.t_conv(h)
        s_cc = self.s_conv(m)
        x_cc = self.x2_conv(x)

        i_s, g_s, f_s, o_s = torch.chunk(s_cc, 4, dim=1)
        i_t, g_t, o_t = torch.chunk(t_cc, 3, dim=1)
        i_x, g_x, f_x, o_x = torch.chunk(x_cc, 4, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)

        new_m = f_ * m + i_ * g_
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c)
        new_c = c + i * g

        cell = torch.cat([new_c, new_m], dim=1)
        cell = self.cell_reduce(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


class MIMN(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden,
                 seq_shape, tln=True):
        super().__init__()

        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.layer_norm = tln
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self._forget_bias = 1.0

        self.h_conv = ConvLN(
            num_hidden, num_hidden * 4,
            filter_size, tln
        )
        self.x_conv = ConvLN(
            num_hidden, num_hidden * 4,
            filter_size, tln
        )

        self.ct_weight = nn.Parameter(
            torch.randn(1, num_hidden * 2, self.height, self.width)
        )
        self.oc_weight = nn.Parameter(
            torch.randn(1, num_hidden, self.height, self.width)
        )

    def init_state(self, device, batch):
        return torch.zeros(
            batch, self.num_hidden, self.height, self.width,
            device=device
        )

    def forward(self, x, h_t, c_t):
        device = x.device if x is not None else h_t.device
        B = x.size(0) if x is not None else h_t.size(0)

        if h_t is None:
            h_t = self.init_state(device, B)
        if c_t is None:
            c_t = self.init_state(device, B)

        h_concat = self.h_conv(h_t)
        i_h, g_h, f_h, o_h = torch.chunk(h_concat, 4, dim=1)

        ct_activation = torch.cat([c_t, c_t], dim=1) * self.ct_weight
        i_c, f_c = torch.chunk(ct_activation, 2, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.x_conv(x)
            i_x, g_x, f_x, o_x = torch.chunk(x_concat, 4, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = c_new * self.oc_weight
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, tln=False):
        super().__init__()

        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self.layer_norm = tln
        self._forget_bias = 1.0

        self.t_conv = ConvLN(
            num_hidden, num_hidden * 4,
            filter_size, tln
        )
        self.s_conv = ConvLN(
            num_hidden, num_hidden * 4,
            filter_size, tln
        )
        self.x_conv = ConvLN(
            num_hidden_in, num_hidden * 4,
            filter_size, tln
        )

        self.cell_reduce = nn.Conv2d(
            num_hidden * 2, num_hidden,
            kernel_size=1
        )

    def init_state(self, device, batch):
        return torch.zeros(
            batch, self.num_hidden,
            self.height, self.width,
            device=device
        )

    def forward(self, x, h, c, m):
        device = x.device
        B = x.size(0)

        if h is None:
            h = self.init_state(device, B)
        if c is None:
            c = self.init_state(device, B)
        if m is None:
            m = self.init_state(device, B)

        t_cc = self.t_conv(h)
        s_cc = self.s_conv(m)
        x_cc = self.x_conv(x)

        i_s, g_s, f_s, o_s = torch.chunk(s_cc, 4, dim=1)
        i_t, g_t, f_t, o_t = torch.chunk(t_cc, 4, dim=1)
        i_x, g_x, f_x, o_x = torch.chunk(x_cc, 4, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)

        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)

        f = torch.sigmoid(f_x + f_t + self._forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)

        o = torch.sigmoid(o_x + o_t + o_s)

        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g

        cell = torch.cat([new_c, new_m], dim=1)
        cell = self.cell_reduce(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


def w_initializer(dim_in, dim_out):
    random_range = math.sqrt(6.0 / (dim_in + dim_out))
    return (-random_range, random_range)



class MIM(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        num_layers=4,
        num_hidden=[64, 64, 64, 64],
        filter_size=5,
        stride=1,
        tln=True
    ):
        super().__init__()

        B, T_in, C_in, H, W = input_shape
        _, T_out, C_out, _, _ = output_shape

        self.T_in = T_in
        self.T_out = T_out
        self.total_length = T_in + T_out
        self.input_length = T_in

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.C_out = C_out

        self.stlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                num_hidden_in = C_in
            else:
                num_hidden_in = num_hidden[i - 1]

            if i < 1:
                layer = SpatioTemporalLSTMCell(
                    f'SpatioTemporalLSTMCell_{i+1}',
                    filter_size,
                    num_hidden_in,
                    num_hidden[i],
                    input_shape,
                    tln=tln
                )
            else:
                layer = MIMBlock(
                    f'SpatioTemporalLSTMCell_{i+1}',
                    filter_size,
                    num_hidden_in,
                    num_hidden[i],
                    input_shape,
                    tln=tln
                )

            self.stlstm_layers.append(layer)

        self.diff_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            layer = MIMN(
                f'SpatioTemporalLSTMCell_diff{i+1}',
                filter_size,
                num_hidden[i + 1],
                input_shape,
                tln=tln
            )
            self.diff_layers.append(layer)

        self.back_to_pixel = nn.Conv2d(
            num_hidden[num_layers - 1],
            C_out,
            kernel_size=1,
            stride=1,
            padding=0
        )

        low, high = w_initializer(num_hidden[num_layers - 1], C_out)
        nn.init.uniform_(self.back_to_pixel.weight, low, high)
        nn.init.zeros_(self.back_to_pixel.bias)

    def forward(self, images, schedual_sampling_bool=None):
        for layer in self.stlstm_layers:
            if isinstance(layer, MIMBlock):
                layer.convlstm_c = None

        device = images.device
        B = images.size(0)

        hidden_state = [None] * self.num_layers
        cell_state = [None] * self.num_layers

        hidden_state_diff = [None] * (self.num_layers - 1)
        cell_state_diff = [None] * (self.num_layers - 1)

        st_memory = None
        gen_images = []

        for time_step in range(self.total_length - 1):

            if time_step < self.input_length:
                x_gen = images[:, time_step]
            else:
                x_gen = gen_images[-1]

            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = \
                self.stlstm_layers[0](
                    x_gen,
                    hidden_state[0],
                    cell_state[0],
                    st_memory
                )

            for i in range(1, self.num_layers):

                if time_step > 0:
                    if i == 1:
                        diff_input = hidden_state[i - 1] - preh
                    else:
                        diff_input = hidden_state_diff[i - 2]

                    hidden_state_diff[i - 1], cell_state_diff[i - 1] = \
                        self.diff_layers[i - 1](
                            diff_input,
                            hidden_state_diff[i - 1],
                            cell_state_diff[i - 1]
                        )
                else:
                    _ = self.diff_layers[i - 1](
                        torch.zeros_like(hidden_state[i - 1]),
                        None,
                        None
                    )

                preh = hidden_state[i]

                hidden_state[i], cell_state[i], st_memory = \
                    self.stlstm_layers[i](
                        hidden_state[i - 1],
                        hidden_state_diff[i - 1],
                        hidden_state[i],
                        cell_state[i],
                        st_memory
                    )

            x_gen = self.back_to_pixel(
                hidden_state[self.num_layers - 1]
            )

            gen_images.append(x_gen)

        gen_images = torch.stack(gen_images, dim=1)
        return gen_images[:, -self.T_out:]
