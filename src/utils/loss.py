import torch
import torch.nn.functional as F

def masked_mse(output, target, nan_mask):
    loss = torch.nn.MSELoss()
    # mask = ~nan_mask  # valid positions
    return loss(output[nan_mask==0], target[nan_mask==0])