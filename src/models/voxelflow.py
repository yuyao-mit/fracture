
# https://arxiv.org/pdf/1702.02463

"""Implements a phase-field prediction model based on voxel flow architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l1_loss(predictions, targets):
    """L1 loss function."""
    return F.l1_loss(predictions, targets)


def l2_loss(predictions, targets):
    """L2 loss function."""
    return F.mse_loss(predictions, targets)


def meshgrid(height, width, device='cuda'):
    """Create meshgrid for coordinate mapping."""
    y_coords = torch.arange(0, height, dtype=torch.float32, device=device)
    x_coords = torch.arange(0, width, dtype=torch.float32, device=device)
    
    # Create meshgrid
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    return grid_x, grid_y


def bilinear_interp(img, x, y):
    """Bilinear interpolation using PyTorch's grid_sample."""
    batch_size, channels, height, width = img.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * x / (width - 1) - 1.0
    y_norm = 2.0 * y / (height - 1) - 1.0
    
    # Stack coordinates [B, H, W, 2]
    grid = torch.stack([x_norm, y_norm], dim=-1)
    
    # Perform interpolation
    output = F.grid_sample(img, grid, mode='bilinear', 
                          padding_mode='border', align_corners=True)
    
    return output


class TemporalFeatureExtractor(nn.Module):
    """Extract temporal features from sequence of phase-field data."""
    
    def __init__(self, input_channels=10, hidden_channels=64):
        super(TemporalFeatureExtractor, self).__init__()
        
        # 3D convolutions for temporal-spatial feature extraction
        self.conv3d_1 = nn.Conv3d(input_channels, hidden_channels, 
                                  kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(hidden_channels)
        
        self.conv3d_2 = nn.Conv3d(hidden_channels, hidden_channels * 2, 
                                  kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(hidden_channels * 2)
        
        # Temporal pooling to reduce T dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] -> [B, C, T, H, W] for conv3d
        """
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        
        # Temporal pooling: [B, C*2, T, H, W] -> [B, C*2, 1, H, W]
        x = self.temporal_pool(x)
        
        # Remove temporal dimension: [B, C*2, H, W]
        x = x.squeeze(2)
        
        return x


class PhaseFieldFlowPredictor(nn.Module):
    def __init__(self, is_train=True, input_channels=10, hidden_dim=128):
        super(PhaseFieldFlowPredictor, self).__init__()
        self.is_train = is_train
        self.input_channels = input_channels
        
        # Temporal feature extractor
        self.temporal_extractor = TemporalFeatureExtractor(input_channels, 64)
        
        # Flow prediction network (modified voxel flow architecture)
        # Input: temporal features (128 channels from temporal extractor)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder with upsampling
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        
        # Output: flow (2 channels) + mask (1 channel) for each input channel
        # Total: (2 + 1) * input_channels = 3 * input_channels
        self.conv7 = nn.Conv2d(64, 3 * input_channels, kernel_size=5, stride=1, padding=2)
        
        # Direct prediction branch for residual learning
        self.direct_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.direct_bn1 = nn.BatchNorm2d(64)
        self.direct_conv2 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input_sequence):
        """Forward pass through the network.
        
        Args:
            input_sequence: Input tensor of shape [B, T=5, C=10, H=64, W=64]
            
        Returns:
            Predicted next frame of shape [B, 1, C=10, H=64, W=64]
        """
        batch_size, T, C, H, W = input_sequence.shape
        device = input_sequence.device
        
        # Extract temporal features
        temporal_features = self.temporal_extractor(input_sequence)  # [B, 128, H, W]
        
        # Flow prediction network
        x = F.relu(self.bn1(self.conv1(temporal_features)))
        x = self.pool1(x)  # [B, 64, H/2, W/2]
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # [B, 128, H/4, W/4]
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # [B, 256, H/8, W/8]
        
        # Decoder path with upsampling
        x = F.interpolate(x, size=(H//4, W//4), mode='bilinear', align_corners=False)
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=False)
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Flow and mask prediction
        flow_mask = torch.tanh(self.conv7(x))  # [B, 3*C, H, W]
        
        # Direct prediction for residual learning
        direct_pred = self.direct_conv1(temporal_features)
        direct_pred = F.relu(self.direct_bn1(direct_pred))
        direct_pred = self.direct_conv2(direct_pred)  # [B, C, H, W]
        
        # Process each channel separately
        predictions = []
        
        for c in range(C):
            # Extract flow and mask for this channel
            flow_c = flow_mask[:, c*3:(c*3+2), :, :]  # [B, 2, H, W]
            mask_c = flow_mask[:, c*3+2:c*3+3, :, :]  # [B, 1, H, W]
            
            # Create coordinate grids
            grid_x, grid_y = meshgrid(H, W, device)
            grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Scale flow
            flow_c = 0.5 * flow_c
            
            # Use the last two frames for warping-based prediction
            frame_t4 = input_sequence[:, -1, c:c+1, :, :]  # [B, 1, H, W]
            frame_t3 = input_sequence[:, -2, c:c+1, :, :]  # [B, 1, H, W]
            
            # Calculate warping coordinates
            coor_x_1 = grid_x + flow_c[:, 0, :, :]
            coor_y_1 = grid_y + flow_c[:, 1, :, :]
            coor_x_2 = grid_x - flow_c[:, 0, :, :]
            coor_y_2 = grid_y - flow_c[:, 1, :, :]
            
            # Warp frames
            warped_1 = bilinear_interp(frame_t4, coor_x_1, coor_y_1)
            warped_2 = bilinear_interp(frame_t3, coor_x_2, coor_y_2)
            
            # Apply mask for blending
            mask_c = 0.5 * (1.0 + mask_c)
            
            # Warped prediction
            warped_pred = mask_c * warped_1 + (1.0 - mask_c) * warped_2
            
            # Combine with direct prediction (residual learning)
            combined_pred = warped_pred + 0.1 * direct_pred[:, c:c+1, :, :]
            
            predictions.append(combined_pred)
        
        # Stack all channel predictions
        result = torch.cat(predictions, dim=1)  # [B, C, H, W]
        
        # Add time dimension: [B, C, H, W] -> [B, 1, C, H, W]
        result = result.unsqueeze(1)
        
        return result
    
    def compute_loss(self, predictions, targets, alpha=0.8):
        """Compute the training loss.
        
        Args:
            predictions: Predicted frames [B, 1, C, H, W]
            targets: Ground truth frames [B, 1, C, H, W]
            alpha: Weight for L1 vs L2 loss combination
            
        Returns:
            Loss value
        """
        l1_loss_val = l1_loss(predictions, targets)
        l2_loss_val = l2_loss(predictions, targets)
        
        # Combine L1 and L2 losses
        total_loss = alpha * l1_loss_val + (1 - alpha) * l2_loss_val
        
        return total_loss