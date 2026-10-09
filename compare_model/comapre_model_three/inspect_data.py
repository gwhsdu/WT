import torch
import sys
import os

print("Loading data...")
wind = torch.load('./data/wind.pt')
aa = torch.load('./data/aa.pt')

print(f"wind shape: {wind.shape}")
print(f"wind dtype: {wind.dtype}")
print(f"aa shape: {aa.shape}")
print(f"aa dtype: {aa.dtype}")

# Check dimensions
N_case, N_time, C, L, W = wind.shape
print(f"N_case={N_case}, N_time={N_time}, C={C}, L={L}, W={W}")
print(f"N_turbine={aa.shape[1]}")

# Print first few values
print("First case aa:", aa[0])
print("Wind stats: mean={:.4f}, std={:.4f}".format(wind.mean().item(), wind.std().item()))

# Check for NaNs
print("Wind NaNs:", torch.isnan(wind).any().item())
print("aa NaNs:", torch.isnan(aa).any().item())

# Check valid time steps
print("Total possible samples per case:", N_time - 1)