import torch
import torch.nn as nn
import numpy as np


class PodMLP(nn.Module):
    """
    POD-MLP baseline.
    """
    def __init__(self, spatial_shape, param_dim=1, n_components=50, hidden_width=128):
        """
        Args:
            spatial_shape: tuple (C, L, W) where C=3
            param_dim: N_turbine
            n_components: number of POD modes to keep (r)
            hidden_width: hidden dimension of MLP (default 128)
        """
        super().__init__()
        self.C, self.L, self.W = spatial_shape
        self.param_dim = param_dim
        self.n_components = n_components
        self.flatten_dim = self.C * self.L * self.W

        # PCA components (will be filled by fit(), registered as buffers)
        self.register_buffer('pca_mean', None)          # (flatten_dim,)
        self.register_buffer('pca_components', None)    # (flatten_dim, n_components)
        self.register_buffer('pca_std', None)           # (n_components,) sqrt of eigenvalues

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_components + param_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, n_components)
        )

    def fit(self, wind_data):
        """
        Fit PCA on flattened wind data.
        Args:
            wind_data: tensor of shape (N, C, L, W)
        """
        N = wind_data.shape[0]
        # Flatten
        X = wind_data.reshape(N, -1)  # (N, D) where D = C*L*W
        # Compute mean
        mean = X.mean(dim=0)  # (D,)
        X_centered = X - mean
        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        # Vt shape (min(N, D), D), we need components as columns of V
        # Keep top n_components
        r = min(self.n_components, S.shape[0])
        self.n_components = r
        components = Vt[:r, :].T  # (D, r)
        # Standard deviation of principal components
        std = S[:r] / np.sqrt(N - 1)
        # Store as buffers
        self.pca_mean = mean
        self.pca_components = components
        self.pca_std = std
        print(f"Fitted PCA with {r} components.")

    def project(self, wind):
        """
        Project wind field to POD coefficients.
        Args:
            wind: tensor of shape (..., C, L, W)
        Returns:
            coeffs: tensor of shape (..., n_components)
        """
        orig_shape = wind.shape
        D = orig_shape[-3] * orig_shape[-2] * orig_shape[-1]
        # Flatten last three dimensions
        flat = wind.reshape(*orig_shape[:-3], D)
        # Center
        centered = flat - self.pca_mean
        # Project
        coeffs = torch.matmul(centered, self.pca_components)  # (..., n_components)
        return coeffs

    def reconstruct(self, coeffs):
        """
        Reconstruct wind field from POD coefficients.
        Args:
            coeffs: tensor of shape (..., n_components)
        Returns:
            wind: tensor of shape (..., C, L, W)
        """
        # Reconstruct flattened
        flat = torch.matmul(coeffs, self.pca_components.T) + self.pca_mean
        # Reshape
        return flat.reshape(*coeffs.shape[:-1], self.C, self.L, self.W)

    def forward(self, x, p):
        """
        Args:
            x: wind field (B, C, L, W)
            p: parameters (B, param_dim)
        Returns:
            predicted wind field (B, C, L, W) (denormalized? Should be normalized)
        """
        # Project input wind
        a_t = self.project(x)  # (B, n_components)
        # Concatenate with parameters
        z = torch.cat([a_t, p], dim=1)  # (B, n_components + param_dim)
        # MLP predicts next coefficients
        a_t1_pred = self.mlp(z)  # (B, n_components)
        # Reconstruct wind field
        x_pred = self.reconstruct(a_t1_pred)  # (B, C, L, W)
        return x_pred

    def load_pca(self, mean, components, std):
        """Load precomputed PCA components."""
        self.pca_mean = mean
        self.pca_components = components
        self.pca_std = std
        self.n_components = components.shape[1]

    def get_pca(self):
        """Return PCA components."""
        return self.pca_mean, self.pca_components, self.pca_std


if __name__ == '__main__':
    # Test with random data
    C, L, W = 3, 81, 33
    param_dim = 1
    model = PodMLP((C, L, W), param_dim, n_components=50)
    # Generate random training data
    train_data = torch.randn(100, C, L, W)
    model.fit(train_data)
    # Test forward
    x = torch.randn(2, C, L, W)
    p = torch.randn(2, param_dim)
    out = model(x, p)
    print(f"Output shape: {out.shape}")
    print(f"Input shape: {x.shape}")