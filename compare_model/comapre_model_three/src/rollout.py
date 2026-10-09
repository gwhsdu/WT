import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders
from utils import set_seed, ensure_dir, create_param_map, concatenate_wind_param
from models.persistence import Persistence
from models.plain_cnn import PlainCNN
from models.unet import UNet
from models.bicnn_style import BiCNNStyle
from models.pod_mlp import PodMLP
from models.fno import FNO2d


def select_rollout_indices(dataset_test, num_samples=100, rollout_steps=120, seed=42):

    set_seed(seed)
    # Collect all valid starting indices
    valid_indices = []
    # Need to know time indices for test split
    split_idx = dataset_test.test_idx
    split_set = set(split_idx.tolist())
    for case in range(dataset_test.N_case):
        for t in split_idx:
            # Check that t, t+1, ..., t+rollout_steps-1 are all in test split
            valid = all((t + k) in split_set for k in range(rollout_steps+1))
            if valid:
                valid_indices.append((case, t))
    # Random sample
    if len(valid_indices) < num_samples:
        print(f"Warning: only {len(valid_indices)} valid starting points, using all.")
        selected = valid_indices
    else:
        selected = random.sample(valid_indices, num_samples)
    return selected



def recursive_predict(model, start_x, start_p, steps=120, device='cpu'):

    x_mean = torch.load("./outputs/data_norm/wind_mean.pt")
    x_std = torch.load("./outputs/data_norm/wind_std.pt")
    p_mean = torch.load("./outputs/data_norm/param_mean.pt")
    p_std = torch.load("./outputs/data_norm/param_std.pt")
    x_mean = x_mean.view(1, -1, 1, 1)
    x_std = x_std.view(1, -1, 1, 1)
    model.eval()
    x = start_x.clone()
    x = (x - x_mean) / x_std
    start_p = (start_p-p_mean)/p_std
    preds = []
    with torch.no_grad():
        for i in range(steps):
            y_pred = model(x.to(device), start_p.to(device))
            preds.append(y_pred.squeeze(0).cpu())
            x = y_pred  # use prediction as next input
    return torch.stack(preds, dim=0)  # (steps, 3, L, W)


def run_rollout_for_model(model_name, model, indices, dataset_test, device,
                          rollout_steps=120, save_dir='./outputs/predictions'):

    ensure_dir(save_dir)
    predictions = []
    for case_idx, start_t in indices:
        # Get initial wind field (normalized)
        start_x = dataset_test.wind[case_idx, start_t].unsqueeze(0)  # (1,3,L,W)
        start_p = dataset_test.params[case_idx].unsqueeze(0)         # (1, N_turbine)
        # Recursive predict
        pred = recursive_predict(model, start_x, start_p, rollout_steps, device)
        predictions.append(pred)  # (steps, 3, L, W)
    # Stack across samples
    predictions = torch.stack(predictions, dim=0)  # (num_samples, steps, 3, L, W)
    # Denormalize to original scale
    predictions_denorm = dataset_test.denormalize_wind(predictions)
    # Save
    if model_name == 'plain_cnn':
        save_path = os.path.join(
            save_dir,
            'Plain CNN_rollout_seed42.pt',
        )
    elif model_name == 'unet':
        save_path = os.path.join(
            save_dir,
            'U-Net_rollout.pt',
        )
    elif model_name == 'bicnn_style':
        save_path = os.path.join(
            save_dir,
            'BiCNN_rollout.pt',
        )
    elif model_name == 'fno':
        save_path = os.path.join(
            save_dir,
            'FNO_rollout_seed42.pt',
        )

    # save_path = os.path.join(save_dir, f'{model_name}_rollout.pt')
    torch.save(predictions_denorm, save_path)
    print(f"Saved {model_name} predictions to {save_path}")
    return predictions_denorm


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint (state_dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Assume checkpoint is directly state_dict
        model.load_state_dict(checkpoint, strict=False)
    return model


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    wind_path = './data/wind.pt'
    param_path = './data/aa.pt'
    # Use batch size 1 for data loading (just need dataset)
    (_, _, test_loader,
     ds_train, ds_val, ds_test,
     wind_mean, wind_std, param_mean, param_std) = create_data_loaders(
        wind_path, param_path, batch_size=1, seed=42)

    # Select rollout indices
    indices = select_rollout_indices(ds_test, num_samples=100, rollout_steps=120, seed=42)
    # Save indices
    ensure_dir('./outputs/predictions/seed42')
    indices_path = './outputs/predictions/rollout_indices.pt'
    torch.save(indices, indices_path)
    print(f"Saved rollout indices to {indices_path}")
    print(f"Selected {len(indices)} starting points.")

    # Load ground truth for these indices
    ground_truth = []
    for case_idx, start_t in indices:
        # Collect wind fields for steps 1..10 (i.e., start_t+1 to start_t+10)
        truth = torch.stack([ds_test.wind[case_idx, start_t + k] for k in range(1, 121)], dim=0)
        ground_truth.append(truth)  # (steps, 3, L, W)
    ground_truth = torch.stack(ground_truth, dim=0)  # (num_samples, steps, 3, L, W)
    #ground_truth_denorm = ds_test.denormalize_wind(ground_truth)
    gt_path = './outputs/predictions/ground_truth_rollout.pt'
    #torch.save(ground_truth_denorm, gt_path)
    torch.save(ground_truth, gt_path)
    print(f"Saved ground truth to {gt_path}")


    checkpoint_dir = './outputs/checkpoints/seed42'


    model_configs = {
        'plain_cnn': {'class': PlainCNN, 'args': {'param_dim': 3, 'channels': [64, 128, 128, 64, 3]}},
        'unet': {'class': UNet, 'args': {'param_dim': 3, 'base_width': 64}},
        'bicnn_style': {'class': BiCNNStyle, 'args': {'param_dim': 3,
                                                      'bg_channels': [64, 128, 128],
                                                      'fg_channels': [64, 128, 128],
                                                      'fusion_channels': [256, 128, 64, 3]}},
        'fno': {'class': FNO2d,'args': {'in_channels': 3,'param_dim': 3,'width': 64,'modes_h': 16,'modes_w': 12,
                'n_layers': 4, 'padding_h': 8, 'padding_w': 4,'use_grid': True,'projection_width': 128,
            },
        },
    }

    plain_cnn = model_configs['plain_cnn']['class'](**model_configs['plain_cnn']['args']).to(device)
    plain_cnn = load_checkpoint(plain_cnn, os.path.join(checkpoint_dir, 'plain_cnn_best.pt'), device)



    fno = model_configs['fno']['class']( **model_configs['fno']['args']).to(device)

    fno = load_checkpoint( fno, os.path.join(checkpoint_dir,'fno_best.pt',),device,)


    # Run rollout for each model
    models = {
        'plain_cnn': plain_cnn,
        # 'unet': unet,
        # 'bicnn_style': bicnn,
        'fno': fno,
    }

    for name, model in models.items():
        print(f"\nRollout for {name}...")
        run_rollout_for_model(name, model, indices, ds_test, device)

    print("\nRollout completed.")


if __name__ == '__main__':
    main()