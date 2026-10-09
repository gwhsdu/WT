import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_param_map, concatenate_wind_param, construct_wake_mask


class BiCNNStyle(nn.Module):
    """
    BiCNN-style dual-branch baseline.
    """
    def __init__(self, in_channels=3, param_dim=1, bg_channels=None, fg_channels=None, fusion_channels=None):
        super().__init__()
        self.param_dim = param_dim
        total_in = in_channels + param_dim

        # Default channel configurations
        if bg_channels is None:
            bg_channels = [32, 64, 64]
        if fg_channels is None:
            fg_channels = [32, 64, 64]
        if fusion_channels is None:
            fusion_channels = [128, 64, 32, in_channels]
        assert len(bg_channels) == 3, "bg_channels must have length 3"
        assert len(fg_channels) == 3, "fg_channels must have length 3"
        assert len(fusion_channels) == 4, "fusion_channels must have length 4"

        # Background branch (static layer names for compatibility)
        self.bg_conv1 = nn.Conv2d(total_in, bg_channels[0], kernel_size=3, padding=1)
        self.bg_relu1 = nn.ReLU()
        self.bg_conv2 = nn.Conv2d(bg_channels[0], bg_channels[1], kernel_size=3, padding=1)
        self.bg_relu2 = nn.ReLU()
        self.bg_conv3 = nn.Conv2d(bg_channels[1], bg_channels[2], kernel_size=3, padding=1)
        self.bg_relu3 = nn.ReLU()

        # Foreground branch (same architecture)
        self.fg_conv1 = nn.Conv2d(total_in, fg_channels[0], kernel_size=3, padding=1)
        self.fg_relu1 = nn.ReLU()
        self.fg_conv2 = nn.Conv2d(fg_channels[0], fg_channels[1], kernel_size=3, padding=1)
        self.fg_relu2 = nn.ReLU()
        self.fg_conv3 = nn.Conv2d(fg_channels[1], fg_channels[2], kernel_size=3, padding=1)
        self.fg_relu3 = nn.ReLU()

        # Fusion head
        fusion_in = bg_channels[-1] + fg_channels[-1]
        self.fusion_conv1 = nn.Conv2d(fusion_in, fusion_channels[0], kernel_size=3, padding=1)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_conv2 = nn.Conv2d(fusion_channels[0], fusion_channels[1], kernel_size=3, padding=1)
        self.fusion_relu2 = nn.ReLU()
        self.fusion_conv3 = nn.Conv2d(fusion_channels[1], fusion_channels[2], kernel_size=3, padding=1)
        self.fusion_relu3 = nn.ReLU()
        self.fusion_conv4 = nn.Conv2d(fusion_channels[2], fusion_channels[3], kernel_size=3, padding=1)
        # No activation after last layer

    def forward(self, x, p):
        B, C, L, W = x.shape
        param_map = create_param_map(p, (L, W))
        # Background input
        bg_inp = concatenate_wind_param(x, param_map)

        # Foreground mask
        mask = construct_wake_mask(x)  # (B, 1, L, W)
        # Multiply mask across channels
        x_fg = x * mask  # (B, 3, L, W)
        fg_inp = concatenate_wind_param(x_fg, param_map)

        # Background branch
        bg = self.bg_conv1(bg_inp)
        bg = self.bg_relu1(bg)
        bg = self.bg_conv2(bg)
        bg = self.bg_relu2(bg)
        bg = self.bg_conv3(bg)
        bg = self.bg_relu3(bg)

        # Foreground branch
        fg = self.fg_conv1(fg_inp)
        fg = self.fg_relu1(fg)
        fg = self.fg_conv2(fg)
        fg = self.fg_relu2(fg)
        fg = self.fg_conv3(fg)
        fg = self.fg_relu3(fg)

        # Concatenate
        combined = torch.cat([bg, fg], dim=1)  # (B, bg_channels[-1] + fg_channels[-1], L, W)

        # Fusion
        out = self.fusion_conv1(combined)
        out = self.fusion_relu1(out)
        out = self.fusion_conv2(out)
        out = self.fusion_relu2(out)
        out = self.fusion_conv3(out)
        out = self.fusion_relu3(out)
        delta = self.fusion_conv4(out)  # (B, in_channels, L, W)

        # Residual
        return x + delta


if __name__ == '__main__':
    model = BiCNNStyle(param_dim=1)
    x = torch.randn(2, 3, 81, 33)
    p = torch.randn(2, 1)
    out = model(x, p)
    print(f"Output shape: {out.shape}")
    print(f"Input shape: {x.shape}")