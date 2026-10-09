import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Two-dimensional Fourier convolution with truncated modes."""

    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if modes_h <= 0 or modes_w <= 0:
            raise ValueError("modes_h and modes_w must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w

        # rfft2 keeps positive frequencies along W, while the H dimension still
        # contains both positive and negative frequencies. Therefore two sets
        # of complex weights are used for the two ends of the H spectrum.
        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes_h, modes_w)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*weight_shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*weight_shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_multiply(x_ft, weight):
        return torch.einsum("bihw,iohw->bohw", x_ft, weight)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        _, _, height, width = x.shape
        max_modes_h = height // 2
        max_modes_w = width // 2 + 1
        if self.modes_h > max_modes_h:
            raise ValueError(
                f"modes_h={self.modes_h} is too large for height={height}; "
                f"it must be <= {max_modes_h}"
            )
        if self.modes_w > max_modes_w:
            raise ValueError(
                f"modes_w={self.modes_w} is too large for width={width}; "
                f"it must be <= {max_modes_w}"
            )

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            height,
            max_modes_w,
            dtype=x_ft.dtype,
            device=x.device,
        )

        weight_positive = self.weight_positive.to(dtype=x_ft.dtype)
        weight_negative = self.weight_negative.to(dtype=x_ft.dtype)
        out_ft[:, :, : self.modes_h, : self.modes_w] = self._complex_multiply(
            x_ft[:, :, : self.modes_h, : self.modes_w], weight_positive
        )
        out_ft[:, :, -self.modes_h :, : self.modes_w] = self._complex_multiply(
            x_ft[:, :, -self.modes_h :, : self.modes_w], weight_negative
        )

        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    """A Fourier layer plus a pointwise spatial-domain linear branch."""

    def __init__(self, width, modes_h, modes_w):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes_h, modes_w)
        self.pointwise_conv = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return self.spectral_conv(x) + self.pointwise_conv(x)


class FNO2d(nn.Module):
    """
    Two-dimensional Fourier Neural Operator for one-step wind prediction.

    The external interface matches the existing CNN models:
        x: normalized wind field [B, in_channels, H, W]
        p: normalized case parameters [B, param_dim]
        output: normalized next-step wind field [B, in_channels, H, W]

    The model expands p to a spatial map, optionally appends normalized grid
    coordinates, predicts a wind-field increment, and returns x + delta.
    """

    def __init__(
        self,
        in_channels=3,
        param_dim=1,
        width=64,
        modes_h=16,
        modes_w=12,
        n_layers=4,
        padding_h=8,
        padding_w=4,
        use_grid=True,
        projection_width=128,
    ):
        super().__init__()
        if in_channels <= 0 or param_dim <= 0:
            raise ValueError("in_channels and param_dim must be positive")
        if width <= 0 or projection_width <= 0:
            raise ValueError("width and projection_width must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if padding_h < 0 or padding_w < 0:
            raise ValueError("padding_h and padding_w must be non-negative")

        self.in_channels = in_channels
        self.param_dim = param_dim
        self.width = width
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.use_grid = use_grid

        lifted_channels = in_channels + param_dim + (2 if use_grid else 0)
        self.lifting = nn.Conv2d(lifted_channels, width, kernel_size=1)
        self.fno_blocks = nn.ModuleList(
            [FNOBlock2d(width, modes_h, modes_w) for _ in range(n_layers)]
        )
        self.projection1 = nn.Conv2d(width, projection_width, kernel_size=1)
        self.projection2 = nn.Conv2d(
            projection_width, in_channels, kernel_size=1
        )

    @staticmethod
    def _make_grid(batch_size, height, width, device, dtype):
        grid_h = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        grid_w = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        grid_h = grid_h.view(1, 1, height, 1).expand(
            batch_size, 1, height, width
        )
        grid_w = grid_w.view(1, 1, 1, width).expand(
            batch_size, 1, height, width
        )
        return torch.cat([grid_h, grid_w], dim=1)

    def forward(self, x, p):
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected x with {self.in_channels} channels, got {x.shape[1]}"
            )
        if p.ndim == 1:
            p = p.unsqueeze(-1)
        if p.ndim != 2:
            raise ValueError(f"p must have shape [B, param_dim], got {p.shape}")
        if p.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]}, p has {p.shape[0]}"
            )
        if p.shape[1] != self.param_dim:
            raise ValueError(
                f"Expected p with {self.param_dim} features, got {p.shape[1]}"
            )

        batch_size, _, height, width = x.shape
        param_map = p.to(device=x.device, dtype=x.dtype).view(
            batch_size, self.param_dim, 1, 1
        ).expand(-1, -1, height, width)
        features = [x, param_map]
        if self.use_grid:
            features.append(
                self._make_grid(batch_size, height, width, x.device, x.dtype)
            )

        z = self.lifting(torch.cat(features, dim=1))
        if self.padding_h or self.padding_w:
            z = F.pad(z, (0, self.padding_w, 0, self.padding_h))

        for index, block in enumerate(self.fno_blocks):
            z = block(z)
            if index < len(self.fno_blocks) - 1:
                z = F.gelu(z)

        if self.padding_h:
            z = z[..., : -self.padding_h, :]
        if self.padding_w:
            z = z[..., : -self.padding_w]

        z = F.gelu(self.projection1(z))
        delta = self.projection2(z)
        return x + delta


if __name__ == "__main__":
    model = FNO2d()
    sample_x = torch.randn(2, 3, 81, 33)
    sample_p = torch.randn(2, 1)
    sample_output = model(sample_x, sample_p)
    print(f"Output shape: {sample_output.shape}")
    print(f"Input shape: {sample_x.shape}")
