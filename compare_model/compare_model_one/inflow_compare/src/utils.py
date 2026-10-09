import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys

def set_seed(seed=10):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(dir_path):
    """Create directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)
    

def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def create_param_map(p, spatial_shape):

    B, param_dim = p.shape
    L, W = spatial_shape
    # Repeat across spatial dimensions
    param_map = p.view(B, param_dim, 1, 1).repeat(1, 1, L, W)
    return param_map


def concatenate_wind_param(wind, param_map):

    return torch.cat([wind, param_map], dim=1)


def extract_inflow(wind_tp1):

    return wind_tp1[:, :, :1, :]


def expand_inflow_to_map(inflow, L):

    return inflow.repeat(1, 1, L, 1)


def expand_status_to_map(status, L, W):

    if status.dim() == 2:
        B, N_turbine = status.shape
        status = status.view(B, N_turbine, 1, 1)
    return status.repeat(1, 1, L, W)


def compute_wind_speed(wind):

    return torch.sqrt(torch.sum(wind**2, dim=-3))