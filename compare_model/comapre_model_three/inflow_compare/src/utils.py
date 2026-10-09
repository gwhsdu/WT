import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys

def set_seed(seed=42):
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


def create_param_map(p, spatial_shape):
    """
    Expand parameter vector to a spatial map.

    Args:
        p: (B, param_dim)
        spatial_shape: (L, W)

    Returns:
        param_map: (B, param_dim, L, W)
    """
    B, param_dim = p.shape
    L, W = spatial_shape
    # Repeat across spatial dimensions
    param_map = p.view(B, param_dim, 1, 1).repeat(1, 1, L, W)
    return param_map


def concatenate_wind_param(wind, param_map):
    """
    Concatenate wind field and parameter map along channel dimension.

    Args:
        wind: (B, 3, L, W)
        param_map: (B, param_dim, L, W)

    Returns:
        concat: (B, 3 + param_dim, L, W)
    """
    return torch.cat([wind, param_map], dim=1)


def extract_inflow(wind_tp1):
    """
    Extract inflow from wind tensor at time t+1.

    Args:
        wind_tp1: (B, 3, L, W)

    Returns:
        inflow: (B, 3, 1, W)
    """
    # inflow is the upstream boundary (first row)
    return wind_tp1[:, :, :1, :]


def expand_inflow_to_map(inflow, L):
    """
    Expand inflow to a spatial map by repeating along the height dimension.

    Args:
        inflow: (B, 3, 1, W)
        L: target height

    Returns:
        inflow_map: (B, 3, L, W)
    """
    return inflow.repeat(1, 1, L, 1)


def expand_status_to_map(status, L, W):
    """
    Expand turbine status to a spatial map.

    Args:
        status: (B, N_turbine) or (B, N_turbine, 1, 1)
        L: height
        W: width

    Returns:
        status_map: (B, N_turbine, L, W)
    """
    if status.dim() == 2:
        B, N_turbine = status.shape
        status = status.view(B, N_turbine, 1, 1)
    return status.repeat(1, 1, L, W)


def compute_wind_speed(wind):
    """
    Compute wind speed magnitude from three components.

    Args:
        wind: (..., 3, L, W)

    Returns:
        speed: (..., L, W)
    """
    return torch.sqrt(torch.sum(wind**2, dim=-3))