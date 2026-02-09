
# https://arxiv.org/abs/1706.03458

from torch import nn
import torch.nn.functional as F
import torch

class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError

def wrap(input, flow):
    from einops import repeat, rearrange
    B, C, H, W = input.size()
    device = input.device
    xx = repeat(torch.arange(0, W, dtype=torch.float32, device=device), 'w -> h w', h=H)
    yy = repeat(torch.arange(0, H, dtype=torch.float32, device=device), 'h -> h w', w=W)
    xx = repeat(xx, 'h w -> b 1 h w', b=B)
    yy = repeat(yy, 'h w -> b 1 h w', b=B)
    grid = torch.cat((xx, yy), 1)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = rearrange(vgrid, 'b c h w -> b h w c')
    output = torch.nn.functional.grid_sample(input, vgrid, align_corners=True, padding_mode='border')
    return output

class BaseConvRNN(nn.Module):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)// self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0

class TrajGRU(BaseConvRNN):
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=torch.tanh):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout
        self.i2h = nn.Conv2d(in_channels=input_channel,
                            out_channels=self._num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=32,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))
        self.flows_conv = nn.Conv2d(in_channels=32,
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))
        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)

    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)
        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    def forward(self, inputs=None, states=None, seq_len=5):
        device = inputs.device if inputs is not None else states.device
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(device)
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)
        else:
            i2h_slice = None
        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            wrapped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                wrapped_data.append(wrap(prev_h, -flow))
            wrapped_data = torch.cat(wrapped_data, dim=1)
            h2h = self.ret(wrapped_data)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.ones_like(prev_h), p=self._zoneout, training=self.training)
                next_h = torch.where(mask == 0.0, prev_h, next_h)
            outputs.append(next_h)
            prev_h = next_h
        return torch.stack(outputs), next_h

class PhaseFieldPredictor(nn.Module):
    def __init__(self, input_channels=10, hidden_dim=64, num_layers=2, 
                 output_channels=10, L=5, zoneout=0.0):
        super(PhaseFieldPredictor, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_dim
            self.encoder_layers.append(
                TrajGRU(input_channel=in_ch,
                       num_filter=hidden_dim,
                       b_h_w=(1, 64, 64),
                       zoneout=zoneout,
                       L=L,
                       act_type=torch.tanh)
            )
        self.predictor = TrajGRU(input_channel=hidden_dim,
                                num_filter=hidden_dim,
                                b_h_w=(1, 64, 64),
                                zoneout=zoneout,
                                L=L,
                                act_type=torch.tanh)
        self.output_conv = nn.Conv2d(hidden_dim, output_channels, 
                                   kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4)
        hidden_states = []
        current_input = x
        for i, encoder in enumerate(self.encoder_layers):
            encoder._batch_size = B
            encoder._state_height = H
            encoder._state_width = W
            layer_output, final_state = encoder(current_input, seq_len=T)
            hidden_states.append(final_state)
            current_input = layer_output
        self.predictor._batch_size = B
        self.predictor._state_height = H
        self.predictor._state_width = W
        predicted_output, _ = self.predictor(inputs=None, 
                                           states=hidden_states[-1], 
                                           seq_len=1)
        predicted_frame = predicted_output[0]
        output = self.output_conv(predicted_frame)
        output = output.unsqueeze(1)
        return output
