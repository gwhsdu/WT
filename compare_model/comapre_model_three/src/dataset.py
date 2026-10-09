import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os


class WindDataset(Dataset):

    def __init__(self, wind_data, param_data, split='train', seed=42,
                 wind_mean=None, wind_std=None, param_mean=None, param_std=None):

        super().__init__()
        self.wind = wind_data
        self.params = param_data
        self.split = split
        self.seed = seed

        N_case, N_time, C, L, W = wind_data.shape
        self.N_case = N_case
        self.N_time = N_time
        self.C = C
        self.L = L
        self.W = W
        self.N_turbine = param_data.shape[1]

        # Split time indices for each case (same for all cases)
        self.train_idx, self.val_idx, self.test_idx = self._time_split(N_time, seed)

        # Collect valid samples (case_idx, time_idx) for each split
        self.samples = self._collect_samples()

        # Normalization stats (if provided)
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.param_mean = param_mean
        self.param_std = param_std

        # Precompute normalized wind and params for efficiency
        self.normalized_wind = None
        self.normalized_params = None
        if wind_mean is not None and wind_std is not None:
            self.normalized_wind = (wind_data - wind_mean.view(1, 1, -1, 1, 1)) / wind_std.view(1, 1, -1, 1, 1)
        if param_mean is not None and param_std is not None:
            self.normalized_params = (param_data - param_mean) / param_std

    def _time_split(self, N_time, seed):

        idx = np.arange(N_time)
        train_end = int(0.7 * N_time)
        val_end = train_end + int(0.15 * N_time)
        # remaining for test
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
        return train_idx, val_idx, test_idx

    def _collect_samples(self):

        samples = []
        split_idx = getattr(self, self.split + '_idx')
        split_set = set(split_idx.tolist())
        for case in range(self.N_case):
            for t in split_idx:
                if t + 1 < self.N_time and (t + 1) in split_set:
                    samples.append((case, t))
        return samples

    def denormalize_wind(self, wind):
        """Denormalize wind field using stored stats."""
        if self.wind_mean is None or self.wind_std is None:
            raise ValueError("Normalization stats not available.")
        return wind * self.wind_std.view(1, -1, 1, 1) + self.wind_mean.view(1, -1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, time_idx = self.samples[idx]
        if self.normalized_wind is not None:
            X = self.normalized_wind[case_idx, time_idx]  # (3, L, W)
            Y = self.normalized_wind[case_idx, time_idx + 1]
        else:
            X = self.wind[case_idx, time_idx]
            Y = self.wind[case_idx, time_idx + 1]
        if self.normalized_params is not None:
            p = self.normalized_params[case_idx]  # (N_turbine)
        else:
            p = self.params[case_idx]
        return X, p, Y


def compute_normalization_stats(train_dataset):

    # Collect all training samples
    wind_all = []
    param_all = []
    for case_idx, time_idx in train_dataset.samples:
        wind_all.append(train_dataset.wind[case_idx, time_idx])  # (3, L, W)
        param_all.append(train_dataset.params[case_idx])         # (N_turbine,)
    wind_stack = torch.stack(wind_all, dim=0)  # (N_train, 3, L, W)
    param_stack = torch.stack(param_all, dim=0)  # (N_train, N_turbine)

    # Compute per-channel mean/std across spatial dimensions and samples
    wind_mean = wind_stack.mean(dim=(0, 2, 3))  # (3,)
    wind_std = wind_stack.std(dim=(0, 2, 3))
    wind_std[wind_std == 0] = 1.0

    # Parameters: per-feature mean/std
    param_mean = param_stack.mean(dim=0)  # (N_turbine,)
    param_std = param_stack.std(dim=0)
    param_std[param_std == 0] = 1.0
    torch.save(wind_mean,'./outputs/data_norm/wind_mean.pt')
    print("wind_mean.shape:",wind_mean.shape)
    torch.save(wind_std,'./outputs/data_norm/wind_std.pt')
    torch.save(param_mean,'./outputs/data_norm/param_mean.pt')
    torch.save(param_std,'./outputs/data_norm/param_std.pt')

    return wind_mean, wind_std, param_mean, param_std


def create_data_loaders(wind_path, param_path, batch_size=300, seed=42):

    wind = torch.load(wind_path)
    params = torch.load(param_path)

    # Create temporary training dataset to compute stats
    temp_train = WindDataset(wind, params, split='train', seed=seed)
    wind_mean, wind_std, param_mean, param_std = compute_normalization_stats(temp_train)

    # Create real datasets with normalization stats
    dataset_train = WindDataset(wind, params, split='train', seed=seed,
                                wind_mean=wind_mean, wind_std=wind_std,
                                param_mean=param_mean, param_std=param_std)
    dataset_val = WindDataset(wind, params, split='val', seed=seed,
                              wind_mean=wind_mean, wind_std=wind_std,
                              param_mean=param_mean, param_std=param_std)
    dataset_test = WindDataset(wind, params, split='test', seed=seed,
                               wind_mean=wind_mean, wind_std=wind_std,
                               param_mean=param_mean, param_std=param_std)

    # Create DataLoaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader,
            dataset_train, dataset_val, dataset_test,
            wind_mean, wind_std, param_mean, param_std)


if __name__ == '__main__':
    # Quick test
    (train_loader, val_loader, test_loader,
     ds_train, ds_val, ds_test, wm, ws, pm, ps) = create_data_loaders(
        './data/wind.pt', './data/aa.pt', batch_size=4)
    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")
    print(f"Test samples: {len(ds_test)}")
    X, p, Y = next(iter(train_loader))
    print(f"Batch X shape: {X.shape}")
    print(f"Batch p shape: {p.shape}")
    print(f"Batch Y shape: {Y.shape}")
    print(f"Wind mean: {wm}")
    print(f"Wind std: {ws}")
    print(f"Param mean: {pm}")
    print(f"Param std: {ps}")