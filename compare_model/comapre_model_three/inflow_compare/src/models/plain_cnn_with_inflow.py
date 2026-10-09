import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_param_map, concatenate_wind_param, expand_inflow_to_map, expand_status_to_map


class PlainCNNWithInflow(nn.Module):
    """
    Two-branch CNN with future inflow.

    Architecture:
        Main branch: wind_t + turbine_status -> 3 conv layers (64,64,64)
        Inflow branch: inflow_tp1 -> 3 conv layers (32,64,64)
        Fusion: concat -> 2 conv layers (64,64) -> output conv (3)
        Residual prediction: wind_t + delta
    """
    def __init__(self, in_channels=3, param_dim=1):
        """
        Args:
            in_channels: wind channels (3)
            param_dim: N_turbine
        """
        super().__init__()
        self.param_dim = param_dim
        # Main branch: processes wind_t + turbine status
        main_in = in_channels + param_dim
        self.main_conv1 = nn.Conv2d(main_in, 64, kernel_size=3, padding=1)
        self.main_relu1 = nn.ReLU()
        self.main_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.main_relu2 = nn.ReLU()
        self.main_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.main_relu3 = nn.ReLU()
        

        # Inflow branch: processes future inflow
        inflow_in = in_channels
        self.inflow_conv1 = nn.Conv2d(inflow_in, 64, kernel_size=3, padding=1)
        self.inflow_relu1 = nn.ReLU()
        self.inflow_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.inflow_relu2 = nn.ReLU()
        self.inflow_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.inflow_relu3 = nn.ReLU()

        # Fusion
        fusion_in = 128 + 128  # main + inflow features
        self.fusion_conv1 = nn.Conv2d(fusion_in, 128, kernel_size=3, padding=1)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fusion_relu2 = nn.ReLU()

        # Output layer
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, inflow, p):
        """
        Args:
            x: wind field at time t (B, 3, L, W)
            inflow: future inflow at time t+1 (B, 3, 1, W)
            p: turbine status (B, param_dim)
        Returns:
            prediction: wind field at time t+1 (B, 3, L, W)
        """
        B, C, L, W = x.shape

        # Expand status to spatial map
        status_map = create_param_map(p, (L, W))  # (B, param_dim, L, W)

        # Expand inflow to spatial map
        inflow_map = expand_inflow_to_map(inflow, L)  # (B, 3, L, W)

        # Main branch input: concatenate wind_t + status_map
        main_input = concatenate_wind_param(x, status_map)  # (B, 3+param_dim, L, W)

        # Main branch forward
        f_main = self.main_conv1(main_input)
        f_main = self.main_relu1(f_main)
        f_main = self.main_conv2(f_main)
        f_main = self.main_relu2(f_main)
        f_main = self.main_conv3(f_main)
        f_main = self.main_relu3(f_main)  # (B, 64, L, W)

        # Inflow branch forward
        f_inflow = self.inflow_conv1(inflow_map)
        f_inflow = self.inflow_relu1(f_inflow)
        f_inflow = self.inflow_conv2(f_inflow)
        f_inflow = self.inflow_relu2(f_inflow)
        f_inflow = self.inflow_conv3(f_inflow)
        f_inflow = self.inflow_relu3(f_inflow)  # (B, 64, L, W)

        # Feature fusion
        f_fused = torch.cat([f_main, f_inflow], dim=1)  # (B, 128, L, W)
        f_fused = self.fusion_conv1(f_fused)
        f_fused = self.fusion_relu1(f_fused)
        f_fused = self.fusion_conv2(f_fused)
        f_fused = self.fusion_relu2(f_fused)  # (B, 64, L, W)

        # Output delta
        delta = self.out_conv(f_fused)  # (B, 3, L, W)

        # Residual prediction
        return x + delta


if __name__ == '__main__':
    # Test forward pass
    model = PlainCNNWithInflow(param_dim=1)
    print(model)

    B, L, W = 4, 81, 33
    x = torch.randn(B, 3, L, W)
    inflow = torch.randn(B, 3, 1, W)
    p = torch.randn(B, 1)

    out = model(x, inflow, p)
    print(f"Input shape: {x.shape}")
    print(f"Inflow shape: {inflow.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape