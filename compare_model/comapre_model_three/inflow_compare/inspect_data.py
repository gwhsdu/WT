import torch
import os

data_dir = './data'
wind = torch.load(os.path.join(data_dir, 'wind.pt'))
aa = torch.load(os.path.join(data_dir, 'aa.pt'))

print('wind shape:', wind.shape)
print('aa shape:', aa.shape)
print('wind dtype:', wind.dtype)
print('aa dtype:', aa.dtype)
print('Number of cases:', wind.shape[0])
print('Number of time steps:', wind.shape[1])
print('Number of turbines:', aa.shape[1])
print('Spatial dimensions L, W:', wind.shape[3], wind.shape[4])