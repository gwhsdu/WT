import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from dataset import WindForecastingDataset
from models.plain_cnn import PlainCNN
from models.plain_cnn_with_inflow import PlainCNNWithInflow

print("Testing dataset...")
wind = torch.load('./data/wind.pt')
aa = torch.load('./data/aa.pt')
print(f"Wind shape: {wind.shape}")
print(f"AA shape: {aa.shape}")

dataset = WindForecastingDataset(wind, aa, split='train', normalize=True)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"wind_t shape: {sample[0].shape}")
print(f"inflow shape: {sample[1].shape}")
print(f"status shape: {sample[2].shape}")
print(f"wind_tp1 shape: {sample[3].shape}")

print("\nTesting baseline model...")
model_b = PlainCNN(param_dim=1)
wind_t, inflow, status, wind_tp1 = sample
wind_t = wind_t.unsqueeze(0)
status = status.unsqueeze(0)
out = model_b(wind_t, status)
print(f"Input shape: {wind_t.shape}")
print(f"Output shape: {out.shape}")
assert out.shape == wind_t.shape

print("\nTesting inflow model...")
model_i = PlainCNNWithInflow(param_dim=1)
inflow = inflow.unsqueeze(0)
out_i = model_i(wind_t, inflow, status)
print(f"Input shape: {wind_t.shape}")
print(f"Inflow shape: {inflow.shape}")
print(f"Output shape: {out_i.shape}")
assert out_i.shape == wind_t.shape

print("\nAll tests passed!")