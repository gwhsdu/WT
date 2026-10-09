import torch
import torch.nn as nn


class Persistence(nn.Module):
    """
    Persistence baseline: predicts Y_{t+1} = X_t.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, p):
        """
        Args:
            x: wind field tensor of shape (B, 3, L, W)
            p: parameter tensor of shape (B, N_turbine) (ignored)
        Returns:
            prediction: same as x
        """
        return x


if __name__ == '__main__':
    model = Persistence()
    x = torch.randn(2, 3, 81, 33)
    p = torch.randn(2, 1)
    out = model(x, p)
    print(f"Output shape: {out.shape}")
    print(torch.allclose(out, x))