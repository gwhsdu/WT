import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WindForecastingDataset(Dataset):

    def __init__(self, wind_tensor, status_tensor, split='train', normalize=True):

        super().__init__()
        self.wind = wind_tensor  # keep original for potential unnormalization
        self.status = status_tensor
        self.split = split
        self.normalize = normalize

        N_case, N_times, C, L, W = wind_tensor.shape
        self.N_case = N_case
        self.N_times = N_times
        self.C = C
        self.L = L
        self.W = W
        self.N_turbine = status_tensor.shape[1]

        # Generate all possible (case, t) pairs
        pairs = []
        for case in range(N_case):
            for t in range(N_times - 1):  # need t+1
                pairs.append((case, t))
        self.pairs = pairs

        # Split along time dimension preserving order
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        # For each case, we split time indices
        train_indices = []
        val_indices = []
        test_indices = []
        for case in range(N_case):
            n = N_times - 1  # number of valid t
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            n_test = n - n_train - n_val
            # start from t=0
            case_train = [(case, t) for t in range(n_train)]
            case_val = [(case, t) for t in range(n_train, n_train + n_val)]
            case_test = [(case, t) for t in range(n_train + n_val, n)]
            train_indices.extend(case_train)
            val_indices.extend(case_val)
            test_indices.extend(case_test)

        if split == 'train':
            self.indices = train_indices
        elif split == 'val':
            self.indices = val_indices
        elif split == 'test':
            self.indices = test_indices
        else:
            raise ValueError(f"Invalid split: {split}")

        # Compute normalization statistics from training set only
        if normalize:
            # Compute mean and std across all training samples (wind components)
            train_wind_samples = []
            for (case, t) in train_indices:
                train_wind_samples.append(wind_tensor[case, t])
            train_wind = torch.stack(train_wind_samples)  # (2*N_train, 3, L, W)
            self.wind_mean = train_wind.mean(dim=(0, 2, 3), keepdim=True)  # (1, 3, 1, 1)
            self.wind_std = train_wind.std(dim=(0, 2, 3), keepdim=True) + 1e-8
            self.wind_std[self.wind_std == 0] = 1.0
            # Normalize all wind data
            self.wind_normalized = (wind_tensor - self.wind_mean) / self.wind_std
            print("wind_tensor.shape", wind_tensor.shape)
            # self.inflow_normalized =(wind_tensor[:,1:,:,:1,:] - wind_tensor[:,:-1,:,:1,:]) / self.wind_std
            
            # ---- status statistics from training set only ----
            status_all = []
            for case, t in train_indices:
                status_all.append(status_tensor[case])   # (N_turbine,)
            
            status_stack = torch.stack(status_all, dim=0)   # (N_train, N_turbine)
            self.status_mean = status_stack.mean(dim=0)     # (N_turbine,)
            self.status_std = status_stack.std(dim=0)       # (N_turbine,)
            self.status_std[self.status_std == 0] = 1.0
            
            self.status_normalized = (status_tensor - self.status_mean) / self.status_std
            
            torch.save(self.wind_mean,'./outputs/data_norm/wind_mean.pt')
            torch.save(self.wind_std,'./outputs/data_norm/wind_std.pt')
            torch.save(self.status_mean,'./outputs/data_norm/param_mean.pt')
            torch.save(self.status_std,'./outputs/data_norm/param_std.pt')
            
        else:
            self.wind_mean = None
            self.wind_std = None
            self.wind_normalized = wind_tensor
            
            self.status_mean = None
            self.status_std = None
            self.status_normalized = status_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        case, t = self.indices[idx]
        wind_t = self.wind_normalized[case, t]          # (3, L, W)
        wind_tp1 = self.wind_normalized[case, t+1]      # (3, L, W)
        inflow_tp1 = wind_tp1[:, :1, :]                 # (3, 1, W)
        # inflow_tp1 = self.inflow_normalized[case, t]
        status = self.status_normalized[case].view(self.N_turbine)  # (N_turbine)

        # Return as tensors
        return wind_t, inflow_tp1, status, wind_tp1

    def get_original_wind(self, case, t):
        """Retrieve original (unnormalized) wind field."""
        return self.wind[case, t]

    def denormalize(self, wind_normalized):
        """Convert normalized wind back to original scale."""
        if self.wind_mean is None:
            return wind_normalized
        return wind_normalized * self.wind_std + self.wind_mean

    def denormalize_wind(self, wind):
        """Alias for compatibility with older code."""
        return self.denormalize(wind)


def create_data_loaders(batch_size=32, shuffle_train=True, num_workers=0):

    wind = torch.load('./data/wind.pt')
    aa = torch.load('./data/aa.pt')

    train_dataset = WindForecastingDataset(wind, aa, split='train', normalize=True)
    val_dataset = WindForecastingDataset(wind, aa, split='val', normalize=True)
    test_dataset = WindForecastingDataset(wind, aa, split='test', normalize=True)

    # Use training statistics for val and test (already normalized in init)
    # Ensure val and test use same mean/std as train
    val_dataset.wind_mean = train_dataset.wind_mean
    val_dataset.wind_std = train_dataset.wind_std
    val_dataset.wind_normalized = (val_dataset.wind - val_dataset.wind_mean) / val_dataset.wind_std
    test_dataset.wind_mean = train_dataset.wind_mean
    test_dataset.wind_std = train_dataset.wind_std
    test_dataset.wind_normalized = (test_dataset.wind - test_dataset.wind_mean) / test_dataset.wind_std
    
    
    val_dataset.status_mean = train_dataset.status_mean
    val_dataset.status_std = train_dataset.status_std
    val_dataset.status_normalized = (val_dataset.status - val_dataset.status_mean) / val_dataset.status_std
    
    test_dataset.status_mean = train_dataset.status_mean
    test_dataset.status_std = train_dataset.status_std
    test_dataset.status_normalized = (test_dataset.status - test_dataset.status_mean) / test_dataset.status_std

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stats = {
        'wind_mean': train_dataset.wind_mean,
        'wind_std': train_dataset.wind_std,
        'N_turbine': train_dataset.N_turbine,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }
    return train_loader, val_loader, test_loader, stats,test_dataset


if __name__ == '__main__':
    # Quick test
    wind = torch.load('./data/wind.pt')
    aa = torch.load('./data/aa.pt')
    print(f"Wind shape: {wind.shape}")
    print(f"Status shape: {aa.shape}")

    dataset = WindForecastingDataset(wind, aa, split='train', normalize=True)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"wind_t shape: {sample[0].shape}")
    print(f"inflow shape: {sample[1].shape}")
    print(f"status shape: {sample[2].shape}")
    print(f"wind_tp1 shape: {sample[3].shape}")