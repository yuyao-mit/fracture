
# https://arxiv.org/pdf/2103.09504

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhaseFieldSpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, height, filter_size, stride, layer_norm):
        super(PhaseFieldSpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.width = width
        self.height = height
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding)
            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding)
            self.conv_a = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding)
            self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding)
            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding)
            
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t, a_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        a_concat = self.conv_a(a_t)
        m_concat = self.conv_m(m_t)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat * a_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Standard LSTM gates
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        # Additional memory gates for phase-field dynamics
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        # Combine memories
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class PhaseFieldPredictor(nn.Module):
    def __init__(self, 
                 input_channels=10,
                 sequence_length=5,
                 img_height=64,
                 img_width=64,
                 num_layers=4,
                 num_hidden=[64, 64, 64, 64],
                 filter_size=5,
                 stride=1,
                 layer_norm=True,
                 decouple_beta=0.01):
        super(PhaseFieldPredictor, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.beta = decouple_beta

        # Loss functions
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()
        
        # Build LSTM cells
        cell_list = []
        
        for i in range(num_layers):
            if i == 0:
                in_channel = num_hidden[0]#input_channels
            else:
                in_channel = num_hidden[i - 1]
            
            cell_list.append(
                PhaseFieldSpatioTemporalLSTMCell(
                    in_channel, num_hidden[i], img_width, img_height,
                    filter_size, stride, layer_norm
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # Output projection layer
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], input_channels, 1, stride=1, padding=0, bias=False)
        
        # Adapter for decoupling loss
        self.adapter = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0, bias=False)
        
        # Multi-scale feature extraction for better phase-field dynamics
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(input_channels, num_hidden[0] // 4, 3, padding=1),
            nn.Conv2d(input_channels, num_hidden[0] // 4, 5, padding=2),
            nn.Conv2d(input_channels, num_hidden[0] // 4, 7, padding=3),
            nn.Conv2d(input_channels, num_hidden[0] // 4, 1, padding=0)
        ])
        
        # Residual connection for phase-field conservation
        self.residual_conv = nn.Conv2d(input_channels, input_channels, 3, padding=1)

    def forward(self, input_sequence, target_sequence=None):
        """
        Args:
            input_sequence: [B, T=5, C=10, H=64, W=64]
            target_sequence: [B, T=1, C=10, H=64, W=64] (for training)
        Returns:
            predictions: [B, T=1, C=10, H=64, W=64]
            loss: scalar (if target_sequence is provided)
        """
        batch_size, seq_len, channels, height, width = input_sequence.shape
        device = input_sequence.device
        
        # Initialize hidden states, cell states, and memory
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros(batch_size, self.num_hidden[i], height, width, device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # Initialize memory for phase-field dynamics
        memory = torch.zeros(batch_size, self.num_hidden[0], height, width, device=device)
        
        decouple_loss = []
        predictions = []

        # Process input sequence
        for t in range(seq_len):
            # Current frame
            current_frame = input_sequence[:, t]  # [B, C, H, W]
            
            # Multi-scale feature extraction for better phase-field representation
            multi_scale_features = []
            for conv in self.multi_scale_conv:
                multi_scale_features.append(conv(current_frame))
            
            # Concatenate multi-scale features
            enhanced_input = torch.cat(multi_scale_features, dim=1)  # [B, num_hidden[0], H, W]
            
            # First LSTM layer with enhanced input
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                enhanced_input, h_t[0], c_t[0], memory, h_t[0]
            )
            
            # Normalize for decoupling loss
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
            )

            # Remaining LSTM layers
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory, h_t[i - 1]
                )
                
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
                )

            # Calculate decoupling loss for each layer
            for i in range(self.num_layers):
                decouple_loss.append(torch.mean(torch.abs(
                    torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                )))

        # Generate next frame prediction
        next_frame = self.conv_last(h_t[self.num_layers - 1])
        
        # Add residual connection for phase-field conservation
        last_frame = input_sequence[:, -1]
        residual = self.residual_conv(last_frame)
        next_frame = next_frame + 0.1 * residual
        
        # Add time dimension: [B, C, H, W] -> [B, 1, C, H, W]
        predictions = next_frame.unsqueeze(1)
        
        # Calculate loss if target is provided
        if target_sequence is not None:
            # Main reconstruction loss
            mse_loss = self.MSE_criterion(predictions, target_sequence)
            l1_loss = self.L1_criterion(predictions, target_sequence)
            
            # Combine losses with phase-field specific weighting
            reconstruction_loss = 0.7 * mse_loss + 0.3 * l1_loss
            
            # Decoupling loss
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
            
            # Total loss
            total_loss = reconstruction_loss + self.beta * decouple_loss
            
            return predictions, total_loss
        
        return predictions