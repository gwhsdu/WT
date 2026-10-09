import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')

from dataset import WindForecastingDataset
from models.plain_cnn import PlainCNN
from models.plain_cnn_with_inflow import PlainCNNWithInflow
from utils import compute_wind_speed, extract_inflow, expand_inflow_to_map


def load_model(checkpoint_path, model_type, param_dim=1, device='cpu'):
    """
    Load trained model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model_type == 'baseline':
        model = PlainCNN(param_dim=param_dim)
    else:
        model = PlainCNNWithInflow(param_dim=param_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model


def compute_inflow_statistics(dataset):
    """
    Compute mean and std of inflow across training dataset.
    """
    inflows = []
    for idx in range(len(dataset)):
        wind_t, inflow, status, wind_tp1 = dataset[idx]
        inflows.append(inflow)
    inflows = torch.stack(inflows)  # (N, 3, 1, W)
    inflow_mean = inflows.mean(dim=(0, 1, 2, 3))
    inflow_std = inflows.std(dim=(0, 1, 2, 3))
    return inflow_mean, inflow_std


def iterative_forecast(model, model_type, wind_0, inflow_sequence, status, device, noise_std=0.0, seed=42):

    if noise_std > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    predictions = []
    current_wind = wind_0.clone()

    for t in range(len(inflow_sequence)):
        inflow = inflow_sequence[t].clone()
        if noise_std > 0 and model_type == 'inflow':
            noise = torch.randn_like(inflow) * noise_std
            inflow = inflow + noise

        with torch.no_grad():
            if model_type == 'baseline':
                pred = model(current_wind, status)
            else:
                pred = model(current_wind, inflow, status)

        predictions.append(pred)
        current_wind = pred  # autoregressive

    return predictions


def compute_relative_error(pred, gt):
    return torch.norm(pred - gt) / torch.norm(gt)


def plot_error_curves(error_dict, save_path):

    plt.figure(figsize=(8, 6))
    for label, errors in error_dict.items():
        steps = np.arange(1, len(errors) + 1)
        plt.plot(steps, errors, marker='o', label=label, linewidth=2)

    plt.xlabel('Iteration Step')
    plt.ylabel('Relative Error')
    plt.title('Iterative Forecast Relative Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_wind_speed(wind, title, save_path, vmin=None, vmax=None):

    speed = compute_wind_speed(wind).squeeze().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(speed.T, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Wind Speed')
    plt.title(title)
    plt.xlabel('L')
    plt.ylabel('W')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths to trained models (adjust as needed)
    baseline_checkpoint = './checkpoints/best_baseline.pt'
    inflow_checkpoint = './checkpoints/best_inflow.pt'

    if not os.path.exists(baseline_checkpoint):
        print(f"Baseline checkpoint not found: {baseline_checkpoint}")
        return
    if not os.path.exists(inflow_checkpoint):
        print(f"Inflow checkpoint not found: {inflow_checkpoint}")
        return

    # Load data
    print("Loading data...")
    wind = torch.load('./data/wind.pt')
    aa = torch.load('./data/aa.pt')

    # Create test dataset (normalized using training stats)
    test_dataset = WindForecastingDataset(wind, aa, split='test', normalize=True)
    # Ensure we have training stats for inflow noise scaling
    train_dataset = WindForecastingDataset(wind, aa, split='train', normalize=True)

    # Compute inflow statistics from training set
    inflow_mean, inflow_std = compute_inflow_statistics(train_dataset)
    print(f"Inflow mean: {inflow_mean:.4f}, std: {inflow_std:.4f}")
    noise_std = 0.05 * inflow_std.item()

    # Choose one test sample (first sample)
    sample_idx = 0
    wind_t, inflow_tp1, status, wind_tp1 = test_dataset[sample_idx]
    # Convert to batch dimension 1
    wind_t = wind_t.unsqueeze(0).to(device)
    inflow_tp1 = inflow_tp1.unsqueeze(0).to(device)
    status = status.unsqueeze(0).to(device)
    wind_tp1 = wind_tp1.unsqueeze(0).to(device)

    # Get original (unnormalized) wind for visualization
    case, t = test_dataset.indices[sample_idx]
    wind_orig = test_dataset.get_original_wind(case, t).unsqueeze(0).to(device)
    wind_tp1_orig = test_dataset.get_original_wind(case, t+1).unsqueeze(0).to(device)

    # We need inflow sequence for 20 steps
    T = 20
    inflow_sequence = []
    # We'll need ground truth wind fields for steps 1..T from the dataset
    # Since we only have one sample, we'll need to get subsequent time steps from same case
    # Ensure we have enough time steps in the test dataset (should be fine)
    gt_winds = [wind_tp1_orig]  # ground truth at step 1
    for i in range(1, T):
        next_t = t + 1 + i
        if next_t < test_dataset.N_times - 1:
            # Get ground truth wind at next step (original scale)
            gt_wind_next = test_dataset.get_original_wind(case, next_t).unsqueeze(0).to(device)
            gt_winds.append(gt_wind_next)
            # Get inflow at next step (normalized)
            # We'll need to extract from normalized wind
            wind_next_norm = test_dataset.wind_normalized[case, next_t].unsqueeze(0).to(device)
            inflow_next = extract_inflow(wind_next_norm)
            inflow_sequence.append(inflow_next)
        else:
            # Not enough steps, repeat last inflow
            inflow_sequence.append(inflow_sequence[-1])
            gt_winds.append(gt_winds[-1])

    # Prepend inflow for first step (already have inflow_tp1)
    inflow_sequence = [inflow_tp1] + inflow_sequence[:T-1]
    gt_winds = gt_winds[:T]  # ensure length T

    # Load models
    print("Loading models...")
    baseline_model = load_model(baseline_checkpoint, 'baseline', param_dim=1, device=device)
    inflow_model = load_model(inflow_checkpoint, 'inflow', param_dim=1, device=device)

    # Run iterative forecasting for each model
    print("Running iterative forecasting...")
    # Baseline (does not use inflow)
    preds_baseline = iterative_forecast(baseline_model, 'baseline', wind_orig, inflow_sequence, status, device, noise_std=0.0)
    # Inflow exact
    preds_inflow_exact = iterative_forecast(inflow_model, 'inflow', wind_orig, inflow_sequence, status, device, noise_std=0.0)
    # Inflow noisy
    preds_inflow_noisy = iterative_forecast(inflow_model, 'inflow', wind_orig, inflow_sequence, status, device, noise_std=noise_std, seed=42)

    # Compute relative errors at each step
    errors_baseline = []
    errors_inflow_exact = []
    errors_inflow_noisy = []

    for step in range(T):
        errors_baseline.append(compute_relative_error(preds_baseline[step], gt_winds[step]).item())
        errors_inflow_exact.append(compute_relative_error(preds_inflow_exact[step], gt_winds[step]).item())
        errors_inflow_noisy.append(compute_relative_error(preds_inflow_noisy[step], gt_winds[step]).item())

    # Print summary
    print("\nRelative Errors after 20 steps:")
    print(f"Baseline: {errors_baseline[-1]:.4f}")
    print(f"Inflow (exact): {errors_inflow_exact[-1]:.4f}")
    print(f"Inflow (noisy): {errors_inflow_noisy[-1]:.4f}")

    # Plot error curves
    error_dict = {
        'Baseline': errors_baseline,
        'Inflow (exact)': errors_inflow_exact,
        'Inflow (noisy)': errors_inflow_noisy
    }
    os.makedirs('./figures', exist_ok=True)
    plot_error_curves(error_dict, './figures/relative_error_vs_iteration.png')
    print("Saved error curve to ./figures/relative_error_vs_iteration.png")

    # Generate wind field visualizations after 20 steps
    # Use the last prediction
    pred_baseline_final = preds_baseline[-1]
    pred_inflow_exact_final = preds_inflow_exact[-1]
    pred_inflow_noisy_final = preds_inflow_noisy[-1]
    gt_final = gt_winds[-1]

    # Determine common color scale
    all_speeds = [
        compute_wind_speed(gt_final).cpu().numpy(),
        compute_wind_speed(pred_baseline_final).cpu().numpy(),
        compute_wind_speed(pred_inflow_exact_final).cpu().numpy(),
        compute_wind_speed(pred_inflow_noisy_final).cpu().numpy()
    ]
    vmin = min([s.min() for s in all_speeds])
    vmax = max([s.max() for s in all_speeds])

    # Plot each
    plot_wind_speed(gt_final, 'Ground Truth (step 20)', './figures/wind_field_ground_truth.png', vmin, vmax)
    plot_wind_speed(pred_baseline_final, 'Baseline Prediction (step 20)', './figures/wind_field_baseline.png', vmin, vmax)
    plot_wind_speed(pred_inflow_exact_final, 'Inflow Exact Prediction (step 20)', './figures/wind_field_inflow_exact.png', vmin, vmax)
    plot_wind_speed(pred_inflow_noisy_final, 'Inflow Noisy Prediction (step 20)', './figures/wind_field_inflow_noisy.png', vmin, vmax)
    print("Saved wind field plots to ./figures/")

    # Save numerical results
    results = {
        'errors_baseline': errors_baseline,
        'errors_inflow_exact': errors_inflow_exact,
        'errors_inflow_noisy': errors_inflow_noisy,
        'noise_std': noise_std,
        'sample_case': case,
        'sample_time_start': t,
        'inflow_mean': inflow_mean.item(),
        'inflow_std': inflow_std.item()
    }
    os.makedirs('./outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'./outputs/iterative_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()