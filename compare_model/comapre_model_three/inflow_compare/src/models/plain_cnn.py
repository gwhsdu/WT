import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_param_map, concatenate_wind_param


class PlainCNN(nn.Module):
    """
    Plain CNN baseline.
    """
    def __init__(self, in_channels=3, param_dim=1, channels=None):
        """
        Args:
            in_channels: wind channels (3)
            param_dim: N_turbine
            channels: list of channel counts for each convolutional layer.
                If None, defaults to [32,64,64,32,in_channels].
        """
        super().__init__()
        self.param_dim = param_dim
        total_in = in_channels + param_dim
        if channels is None:
            channels = [32, 64, 64, 32, in_channels]
        assert len(channels) == 5, "channels must have length 5"
        # Keep original layer names for compatibility with existing checkpoints
        self.conv1 = nn.Conv2d(total_in, channels[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1)

    def forward(self, x, p):
        """
        Args:
            x: wind field (B, 3, L, W)
            p: parameters (B, param_dim)
        Returns:
            residual prediction: x + delta
        """
        B, C, L, W = x.shape
        # Create parameter map
        param_map = create_param_map(p, (L, W))  # (B, param_dim, L, W)
        # Concatenate
        inp = concatenate_wind_param(x, param_map)  # (B, 3+param_dim, L, W)
        # Forward
        out = self.conv1(inp)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        delta = self.conv5(out)  # (B, in_channels, L, W)
        # Residual
        return x + delta


if __name__ == '__main__':
    model = PlainCNN(param_dim=1)
    x = torch.randn(2, 3, 81, 33)
    p = torch.randn(2, 1)
    out = model(x, p)
    print(f"Output shape: {out.shape}")
    print(f"Input shape: {x.shape}")