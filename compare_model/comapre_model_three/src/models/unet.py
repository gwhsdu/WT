import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_param_map, concatenate_wind_param


class DoubleConv(nn.Module):
    """(Conv -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upsampling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use bilinear interpolation for upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """x1 is from decoder, x2 is skip connection"""
        x1 = self.up(x1)
        # Pad if sizes mismatch due to odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Small U-Net baseline.
    """
    def __init__(self, in_channels=3, param_dim=1, base_width=32):
        """
        Args:
            in_channels: wind channels (3)
            param_dim: N_turbine
            base_width: base channel width for the first layer (must be 16, 32, or 64)
        """
        super().__init__()
        self.param_dim = param_dim
        total_in = in_channels + param_dim

        # Encoder
        self.inc = DoubleConv(total_in, base_width)
        self.down1 = Down(base_width, base_width * 2)
        self.down2 = Down(base_width * 2, base_width * 4)
        self.down3 = Down(base_width * 4, base_width * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_width * 8, base_width * 8)

        # Decoder
        self.up1 = Up(base_width * 8 + base_width * 4, base_width * 4)
        self.up2 = Up(base_width * 4 + base_width * 2, base_width * 2)
        self.up3 = Up(base_width * 2 + base_width, base_width)

        # Final 1x1 conv to 3 channels
        self.outc = nn.Conv2d(base_width, in_channels, kernel_size=1)

    def forward(self, x, p):
        """
        Args:
            x: wind field (B, 3, L, W)
            p: parameters (B, param_dim)
        Returns:
            residual prediction: x + delta
        """
        B, C, L, W = x.shape
        # Create parameter map and concatenate
        param_map = create_param_map(p, (L, W))
        inp = concatenate_wind_param(x, param_map)

        # Encoder
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Decoder
        u = self.up1(x5, x3)
        u = self.up2(u, x2)
        u = self.up3(u, x1)

        # Output
        delta = self.outc(u)  # (B, 3, L, W)

        # Residual
        return x + delta


if __name__ == '__main__':
    model = UNet(param_dim=1)
    x = torch.randn(2, 3, 81, 33)
    p = torch.randn(2, 1)
    out = model(x, p)
    print(f"Output shape: {out.shape}")
    print(f"Input shape: {x.shape}")