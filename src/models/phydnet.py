
# https://arxiv.org/pdf/2003.01460

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhaseFieldPhyCell_Cell(nn.Module):
    """Physical dynamics cell specifically designed for phase-field evolution."""
    
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhaseFieldPhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # Physical dynamics function F - models phase-field evolution
        self.F = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, 
                     kernel_size=self.kernel_size, stride=(1,1), padding=self.padding),
            nn.GroupNorm(min(8, F_hidden_dim//4), F_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=F_hidden_dim, out_channels=F_hidden_dim//2, 
                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GroupNorm(min(8, F_hidden_dim//8), F_hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=F_hidden_dim//2, out_channels=input_dim, 
                     kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )

        # Adaptive gate for phase-field correction
        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                 out_channels=self.input_dim,
                                 kernel_size=(3,3),
                                 padding=(1,1), bias=self.bias)
        
        # Phase-field specific constraints
        self.constraint_conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)

    def forward(self, x, hidden):
        if x is not None:
            combined = torch.cat([x, hidden], dim=1)
            combined_conv = self.convgate(combined)
            K = torch.sigmoid(combined_conv)
            
            # Physical prediction
            hidden_tilde = hidden + self.F(hidden)
            
            # Correction with input information
            next_hidden = hidden_tilde + K * (x - hidden_tilde)
        else:
            # In decoding phase, only use physical dynamics
            next_hidden = hidden + self.F(hidden)
        
        # Apply phase-field constraints (optional)
        next_hidden = next_hidden + 0.1 * self.constraint_conv(next_hidden)
        
        return next_hidden


class PhaseFieldPhyCell(nn.Module):
    """Multi-layer physical dynamics cell for phase-field systems."""
    
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhaseFieldPhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhaseFieldPhyCell_Cell(
                input_dim=input_dim,
                F_hidden_dim=self.F_hidden_dims[i],
                kernel_size=self.kernel_size
            ))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_, first_timestep=False):
        if input_ is not None:
            batch_size = input_.data.size()[0]
            if first_timestep:   
                self.initHidden(batch_size)
        elif first_timestep:
            raise ValueError("Cannot initialize hidden states without input in first timestep")
              
        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                if input_ is not None:
                    self.H[j] = cell(input_, self.H[j])
                else:
                    # In decoding phase, use previous hidden state as input
                    self.H[j] = cell(self.H[j], self.H[j])
            else:
                self.H[j] = cell(self.H[j-1], self.H[j])
        
        return self.H, self.H
    
    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.input_dim, 
                                    self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class MultiChannelConvLSTM_Cell(nn.Module):
    """ConvLSTM cell adapted for multi-channel phase-field data."""
    
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        super(MultiChannelConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=4 * self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)
                 
    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        
        if x is not None:
            combined = torch.cat([x, h_cur], dim=1)
            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
        else:
            # In decoding phase, use self-recurrent connections
            combined = torch.cat([h_cur, h_cur], dim=1)
            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class MultiChannelConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for phase-field residual dynamics."""
    
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(MultiChannelConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(MultiChannelConvLSTM_Cell(
                input_shape=self.input_shape,
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_size
            ))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_, first_timestep=False):
        if input_ is not None:
            batch_size = input_.data.size()[0]
            if first_timestep:   
                self.initHidden(batch_size)
        elif first_timestep:
            raise ValueError("Cannot initialize hidden states without input in first timestep")
              
        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                if input_ is not None:
                    self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
                else:
                    # In decoding phase, use previous hidden state as input
                    self.H[j], self.C[j] = cell(self.H[j], (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1], (self.H[j], self.C[j]))
        
        return (self.H, self.C), self.H
    
    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], 
                                    self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], 
                                    self.input_shape[0], self.input_shape[1]).to(self.device))
    
    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


# Enhanced encoder/decoder modules for multi-channel phase-field data
class PhaseFieldEncoder(nn.Module):
    """General encoder for phase-field data: C x H x W -> C_out x H//4 x W//4"""
    
    def __init__(self, nc=10, nf=64):
        super(PhaseFieldEncoder, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, nf//2), nf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        h1 = self.c1(input)   # H/2 x W/2
        h2 = self.c2(h1)      # H/2 x W/2
        h3 = self.c3(h2)      # H/4 x W/4
        return h3


class PhaseFieldDecoder(nn.Module):
    """General decoder for phase-field data: C_in x H//4 x W//4 -> C x H x W"""
    
    def __init__(self, nc=10, nf=64):
        super(PhaseFieldDecoder, self).__init__()
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc3 = nn.ConvTranspose2d(nf, nc, kernel_size=4, stride=2, padding=1)

    def forward(self, input):      
        d1 = self.upc1(input)  # H/2 x W/2
        d2 = self.upc2(d1)     # H/2 x W/2
        d3 = self.upc3(d2)     # H x W
        return d3  


class SpecificEncoder(nn.Module):
    """Specific encoder for physical/residual features."""
    
    def __init__(self, nc=128, nf=128):
        super(SpecificEncoder, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2


class SpecificDecoder(nn.Module):
    """Specific decoder for physical/residual features."""
    
    def __init__(self, nc=128, nf=128):
        super(SpecificDecoder, self).__init__()
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nf//4), nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, nc//4), nc),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)  
        return d2       


class PhaseFieldPhyDNet(nn.Module):
    """Complete PhyDNet model for phase-field prediction."""
    
    def __init__(self, 
                 input_channels=10,
                 input_shape=(16, 16),  # After encoding: 64//4 = 16
                 phycell_hidden_dims=[128, 128],
                 convlstm_hidden_dims=[128, 128],
                 kernel_size=(3, 3),
                 device='cuda'):
        super(PhaseFieldPhyDNet, self).__init__()
        
        self.input_channels = input_channels
        self.device = device
        
        # Encoders and Decoders
        self.encoder_E = PhaseFieldEncoder(nc=input_channels, nf=64)
        self.encoder_Ep = SpecificEncoder(nc=128, nf=128)  # for physics
        self.encoder_Er = SpecificEncoder(nc=128, nf=128)  # for residual
        self.decoder_Dp = SpecificDecoder(nc=128, nf=128)  # for physics
        self.decoder_Dr = SpecificDecoder(nc=128, nf=128)  # for residual  
        self.decoder_D = PhaseFieldDecoder(nc=input_channels, nf=64)

        # Physical and Residual cells
        self.phycell = PhaseFieldPhyCell(
            input_shape=input_shape,
            input_dim=128,
            F_hidden_dims=phycell_hidden_dims,
            n_layers=len(phycell_hidden_dims),
            kernel_size=kernel_size,
            device=device
        )
        
        self.convcell = MultiChannelConvLSTM(
            input_shape=input_shape,
            input_dim=128,
            hidden_dims=convlstm_hidden_dims,
            n_layers=len(convlstm_hidden_dims),
            kernel_size=kernel_size,
            device=device
        )

    def forward(self, input, first_timestep=False, decoding=False):
        """
        Args:
            input: [B, C, H, W] - single time step (None in decoding phase)
            first_timestep: bool - whether to initialize hidden states
            decoding: bool - whether in decoding phase (input=None)
        """
        if not decoding and input is not None:
            input = self.encoder_E(input)  # [B, C, H, W] -> [B, 128, H//4, W//4]
            input_phys = self.encoder_Ep(input)  # Physical features
            input_conv = self.encoder_Er(input)  # Residual features
        else:
            input_phys = None
            input_conv = None

        # Forward through cells
        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        # Decode features
        decoded_Dp = self.decoder_Dp(output1[-1])  # Physical reconstruction
        decoded_Dr = self.decoder_Dr(output2[-1])  # Residual reconstruction
        
        # Individual reconstructions for analysis
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        # Combined reconstruction
        concat = decoded_Dp + decoded_Dr   
        output_image = torch.sigmoid(self.decoder_D(concat))
        
        return output_image, out_phys, out_conv, hidden1, hidden2


class PhaseFieldPredictor(nn.Module):
    """Main predictor for phase-field evolution."""
    
    def __init__(self, 
                 input_channels=10,
                 sequence_length=5,
                 img_height=64,
                 img_width=64,
                 phycell_hidden_dims=[128, 128],
                 convlstm_hidden_dims=[128, 128],
                 device='cpu'):
        super(PhaseFieldPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.device = device
        
        # Calculate encoded shape
        encoded_height = img_height // 4
        encoded_width = img_width // 4
        
        self.phydnet = PhaseFieldPhyDNet(
            input_channels=input_channels,
            input_shape=(encoded_height, encoded_width),
            phycell_hidden_dims=phycell_hidden_dims,
            convlstm_hidden_dims=convlstm_hidden_dims,
            device=device
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, input_sequence, target_sequence=None, predict_steps=1):
        """
        Args:
            input_sequence: [B, T=5, C=10, H=64, W=64]
            target_sequence: [B, T=1, C=10, H=64, W=64] (optional)
            predict_steps: number of steps to predict
        """
        batch_size, seq_len, channels, height, width = input_sequence.shape
        
        predictions = []
        phys_predictions = []
        conv_predictions = []
        
        # Encode input sequence
        for t in range(seq_len):
            current_frame = input_sequence[:, t]  # [B, C, H, W]
            first_timestep = (t == 0)
            
            output_image, out_phys, out_conv, hidden1, hidden2 = self.phydnet(
                current_frame, first_timestep=first_timestep, decoding=False
            )
        
        # Predict future steps
        for t in range(predict_steps):
            output_image, out_phys, out_conv, hidden1, hidden2 = self.phydnet(
                None, first_timestep=False, decoding=True
            )
            
            predictions.append(output_image)
            phys_predictions.append(out_phys)
            conv_predictions.append(out_conv)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [B, predict_steps, C, H, W]
        phys_predictions = torch.stack(phys_predictions, dim=1)
        conv_predictions = torch.stack(conv_predictions, dim=1)
        
        if target_sequence is not None:
            # Calculate losses
            recon_loss = self.mse_loss(predictions, target_sequence)
            
            # Disentanglement loss (encourage physics and residual to be different)
            phys_flat = phys_predictions.view(batch_size, -1)
            conv_flat = conv_predictions.view(batch_size, -1)
            
            # Cosine similarity loss (minimize similarity)
            cos_sim = F.cosine_similarity(phys_flat, conv_flat, dim=1)
            disentangle_loss = torch.mean(torch.abs(cos_sim))
            
            total_loss = recon_loss + 0.01 * disentangle_loss
            
            return predictions, total_loss, {
                'recon_loss': recon_loss,
                'disentangle_loss': disentangle_loss,
                'phys_pred': phys_predictions,
                'conv_pred': conv_predictions
            }
        
        return predictions

