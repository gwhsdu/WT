import torch
import numpy as np
import random
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


def create_param_map(params, spatial_shape):

    B, N = params.shape
    L, W = spatial_shape
    # Reshape to (B, N, 1, 1) and repeat
    param_map = params.view(B, N, 1, 1).expand(-1, -1, L, W)
    return param_map


def concatenate_wind_param(wind, param_map):

    return torch.cat([wind, param_map], dim=1)


def compute_velocity_magnitude(wind):

    return torch.sqrt(torch.sum(wind ** 2, dim=-3))


def construct_wake_mask(wind):

    B, C, L, W = wind.shape
    # Compute velocity magnitude per pixel
    vel_mag = compute_velocity_magnitude(wind)  # (B, L, W)

    # Compute 95th percentile per sample
    # Flatten spatial dimensions
    flat = vel_mag.view(B, -1)  # (B, L*W)
    V_ref = torch.quantile(flat, 0.95, dim=1, keepdim=True)  # (B, 1)
    V_ref = V_ref.view(B, 1, 1)  # (B, 1, 1)

    # Deficit map
    D = V_ref - vel_mag  # (B, L, W)

    # Threshold as 70th percentile of D per sample
    flat_D = D.view(B, -1)
    tau = torch.quantile(flat_D, 0.70, dim=1, keepdim=True)  # (B, 1)
    tau = tau.view(B, 1, 1)

    # Binary mask
    mask = (D > tau).float()  # (B, L, W)
    # Add channel dimension
    mask = mask.unsqueeze(1)  # (B, 1, L, W)
    return mask


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    print("Seed set.")

    # Test param map
    params = torch.randn(2, 5)
    param_map = create_param_map(params, (10, 20))
    print(f"Param map shape: {param_map.shape}")

    # Test concatenate
    wind = torch.randn(2, 3, 10, 20)
    concat = concatenate_wind_param(wind, param_map)
    print(f"Concatenated shape: {concat.shape}")

    # Test mask construction
    mask = construct_wake_mask(wind)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")