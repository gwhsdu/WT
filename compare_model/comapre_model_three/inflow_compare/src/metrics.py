import torch
import numpy as np


def compute_mae(pred, target):
    """
    Compute Mean Absolute Error.
    Args:
        pred: tensor of any shape
        target: same shape as pred
    Returns:
        mae: scalar
    """
    return torch.abs(pred - target).mean().item()


def compute_rmse(pred, target):
    """
    Compute Root Mean Square Error.
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_relative_l2(pred, target, eps=1e-8):
    """
    Compute relative L2 error per sample, then average.
    Args:
        pred: tensor of shape (N, ...)
        target: same shape
        eps: small constant for numerical stability
    Returns:
        relative_l2: scalar
    """
    # Flatten spatial and channel dimensions
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    # Compute per-sample relative L2
    numerator = torch.norm(pred_flat - target_flat, dim=1)  # (N,)
    denominator = torch.norm(target_flat, dim=1) + eps      # (N,)
    rel_l2_per_sample = numerator / denominator  # (N,)
    # Average across samples
    return rel_l2_per_sample.mean().item()


def compute_pointwise_relative_per_step(predictions, ground_truth, eps=1e-8):
    """
    Compute pointwise relative error for each rollout step.
    Defined as |Y_hat - Y| / (|Y| + epsilon) elementwise, then average over channel and spatial dimensions per sample per step, then average over samples.
    Args:
        predictions: tensor of shape (N_samples, rollout_steps, C, L, W)
        ground_truth: same shape
        eps: small constant for numerical stability
    Returns:
        pointwise_rel_per_step: array of length rollout_steps
    """
    N, steps = predictions.shape[:2]
    pointwise_rel_per_step = []
    for s in range(steps):
        pred_s = predictions[:, s]  # (N, C, L, W)
        truth_s = ground_truth[:, s]
        # Elementwise absolute difference
        diff = torch.abs(pred_s - truth_s)  # (N, C, L, W)
        # Elementwise denominator |Y| + eps
        denominator = torch.abs(truth_s) + eps  # (N, C, L, W)
        # Pointwise relative error
        pointwise_rel = diff / denominator  # (N, C, L, W)
        # Average over channel and spatial dimensions per sample
        pointwise_rel_per_sample = pointwise_rel.mean(dim=(1, 2, 3))  # (N,)
        # Average over samples
        pointwise_rel_per_step.append(pointwise_rel_per_sample.mean().item())
    return np.array(pointwise_rel_per_step)


def compute_metrics_per_step_samplewise(predictions, ground_truth):
    """
    Compute MAE, RMSE, and relative L2 for each rollout step, using sample-wise first then average.
    Args:
        predictions: tensor of shape (N_samples, rollout_steps, C, L, W)
        ground_truth: same shape
    Returns:
        mae_per_step: array of length rollout_steps
        rmse_per_step: array of length rollout_steps
        rel_l2_per_step: array of length rollout_steps
    """
    N, steps = predictions.shape[:2]
    #print("predictions.shape:",predictions.shape)
    predictions = torch.sqrt(torch.sum(predictions**2,dim=2,keepdim=True))
    ground_truth = torch.sqrt(torch.sum(ground_truth ** 2, dim=2, keepdim=True))
    #print("predictions_sqrt.shape:",predictions.shape)
    mae_per_step = []
    rmse_per_step = []
    rel_l2_per_step = []
    rel_error_per_step = []
    for s in range(steps):
        pred_s = predictions[:, s]  # (N, C, L, W)
        truth_s = ground_truth[:, s]
        # Compute per-sample MAE
        # Flatten spatial and channel dimensions per sample
        pred_flat = pred_s.reshape(N, -1)
        truth_flat = truth_s.reshape(N, -1)
        mae_per_sample = torch.abs(pred_flat - truth_flat).mean(dim=1)  # (N,)
        mae = mae_per_sample.mean().item()
        mae_per_step.append(mae)
        # Compute per-sample RMSE
        mse_per_sample = torch.mean((pred_flat - truth_flat) ** 2, dim=1)  # (N,)
        rmse_per_sample = torch.sqrt(mse_per_sample)  # (N,)
        rmse = rmse_per_sample.mean().item()
        rmse_per_step.append(rmse)
        # Compute per-sample relative L2
        numerator = torch.norm(pred_flat - truth_flat, dim=1)  # (N,)
        denominator = torch.norm(truth_flat, dim=1) + 1e-8
        rel_l2_per_sample = numerator / denominator
        rel_l2 = rel_l2_per_sample.mean().item()
        rel_l2_per_step.append(rel_l2)
        # Compute per-sample relative error
        abs_error = torch.abs(pred_flat - truth_flat)
        abs_true = torch.abs(truth_flat)
        rel_error_per_sample = abs_error/abs_true
        rel_error = rel_error_per_sample.mean().item()
        rel_error_per_step.append(rel_error)

    return np.array(mae_per_step), np.array(rmse_per_step), np.array(rel_l2_per_step), np.array(rel_error_per_step)


def compute_metrics_per_step(predictions, ground_truth):
    """
    Compute MAE and RMSE for each rollout step (sample-wise first, then average).
    Compatible wrapper that returns only MAE and RMSE.
    """
    # mae, rmse, _ ,_= compute_metrics_per_step_samplewise(predictions, ground_truth)
    _, rmse, _, rel_error = compute_metrics_per_step_samplewise(predictions, ground_truth)
    return rel_error, rmse


def compute_metrics_at_horizons(predictions, ground_truth, horizons=[1,5,10], eps=1e-8):
    """
    Compute MAE, RMSE, relative L2, and pointwise relative error at specific horizons, using sample-wise first.
    Args:
        predictions, ground_truth: as above
        horizons: list of step indices (1-indexed)
        eps: small constant for numerical stability
    Returns:
        dict with keys 'mae@1', 'rmse@1', 'rel_l2@1', 'pointwise_rel@1', etc.
    """
    steps = predictions.shape[1]
    results = {}
    N = predictions.shape[0]
    for h in horizons:
        if h > steps:
            continue
        pred_h = predictions[:, h-1]  # (N, C, L, W)
        truth_h = ground_truth[:, h-1]
        # 合并成模长
        pred_h = torch.sqrt(torch.sum(pred_h ** 2, dim=1, keepdim=True))
        truth_h = torch.sqrt(torch.sum(truth_h ** 2, dim=1, keepdim=True))
        # Flatten per sample
        pred_flat = pred_h.reshape(N, -1)
        truth_flat = truth_h.reshape(N, -1)
        # MAE per sample then average
        mae_per_sample = torch.abs(pred_flat - truth_flat).mean(dim=1)
        results[f'mae@{h}'] = mae_per_sample.mean().item()
        # RMSE per sample then average
        mse_per_sample = torch.mean((pred_flat - truth_flat) ** 2, dim=1)
        rmse_per_sample = torch.sqrt(mse_per_sample)
        results[f'rmse@{h}'] = rmse_per_sample.mean().item()
        # Relative L2 per sample then average
        numerator = torch.norm(pred_flat - truth_flat, dim=1)
        denominator = torch.norm(truth_flat, dim=1) + eps
        rel_l2_per_sample = numerator / denominator
        results[f'rel_l2@{h}'] = rel_l2_per_sample.mean().item()
        # relative error per sample then average
        diff = torch.abs(pred_flat - truth_flat)  # (N, C, L, W)
        denom = torch.abs(truth_flat) + eps   # (N, C, L, W)
        pointwise_rel_per_sample = (diff / denom).mean(dim= 1)  # (N,)
        results[f'pointwise_rel@{h}'] = pointwise_rel_per_sample.mean().item()
    return results
def compute_metrics_per_step_ci(
    predictions,
    ground_truth,
    eps=1e-8,
):
    """
    Compute mean RMSE, mean relative error and their 95% confidence
    intervals across samples for every rollout step.

    Returns:
        rel_mean, rmse_mean,
        rel_lower, rel_upper,
        rmse_lower, rmse_upper
    """

    # Convert three wind components to wind-speed magnitude.
    predictions_speed = torch.sqrt(
        torch.sum(predictions ** 2, dim=2, keepdim=True)
    )

    ground_truth_speed = torch.sqrt(
        torch.sum(ground_truth ** 2, dim=2, keepdim=True)
    )

    difference = predictions_speed - ground_truth_speed

    # Shape: (N_samples, rollout_steps)
    rmse_per_sample = torch.sqrt(
        torch.mean(
            difference ** 2,
            dim=(2, 3, 4),
        )
    )

    rel_error_per_sample = torch.mean(
        torch.abs(difference)
        / (torch.abs(ground_truth_speed) + eps),
        dim=(2, 3, 4),
    )

    def mean_and_ci(values):
        mean = values.mean(dim=0)

        if values.shape[0] > 1:
            std = values.std(dim=0, unbiased=True)
            standard_error = std / np.sqrt(values.shape[0])
            margin = 1.96 * standard_error
        else:
            margin = torch.zeros_like(mean)

        lower = torch.clamp(mean - margin, min=0.0)
        upper = mean + margin

        return (
            mean.cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
        )

    rmse_mean, rmse_lower, rmse_upper = mean_and_ci(
        rmse_per_sample
    )

    rel_mean, rel_lower, rel_upper = mean_and_ci(
        rel_error_per_sample
    )

    return (
        rel_mean,
        rmse_mean,
        rel_lower,
        rel_upper,
        rmse_lower,
        rmse_upper,
    )



# if __name__ == '__main__':
#     # Test
#     pred = torch.randn(10, 10, 3, 81, 33)
#     truth = torch.randn(10, 10, 3, 81, 33)
#     mae_per, rmse_per = compute_metrics_per_step(pred, truth)
#     print(f"MAE per step: {mae_per}")
#     print(f"RMSE per step: {rmse_per}")
#     horizons = compute_metrics_at_horizons(pred, truth, [1,5,10])
#     print("Horizon metrics:", horizons)