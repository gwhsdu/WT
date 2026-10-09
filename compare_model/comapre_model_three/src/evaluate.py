import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_metrics_per_step_samplewise, compute_metrics_at_horizons, compute_pointwise_relative_per_step


def load_predictions(model_names, prediction_dir='./outputs/predictions'):

    preds = {}
    for name in model_names:
        path = os.path.join(prediction_dir, f'{name}_rollout.pt')
        if os.path.exists(path):
            preds[name] = torch.load(path)
        else:
            print(f"Warning: {path} not found.")
    return preds


def evaluate_all():
    prediction_dir = './outputs/predictions'
    table_dir = './outputs/tables'
    os.makedirs(table_dir, exist_ok=True)

    #model_names = ['PPNN-cs', 'Plain CNN', 'U-Net', 'BiCNN','FNO']
    model_names = ['PPNN-cs', 'PPNN-cs_seed20', 'PPNN-cs_seed42', 'Plain CNN', 'Plain CNN_seed20', 'Plain CNN_seed42','FNO','FNO_seed20','FNO_seed42']
    # Load ground truth
    gt_path = os.path.join(prediction_dir, 'ground_truth_rollout.pt')
    ground_truth = torch.load(gt_path)  # (100, 10, 3, L, W)

    # Load predictions
    predictions = load_predictions(model_names, prediction_dir)
    # Ensure all have same shape
    for name, pred in predictions.items():
        assert pred.shape == ground_truth.shape, f"{name} shape mismatch"

    # Compute per-step metrics for each model
    per_step_results = {}
    horizon_results = {}
    steps = ground_truth.shape[1]
    for name in model_names:
        if name not in predictions:
            continue
        pred = predictions[name]
        mae_per, rmse_per, rel_l2_per, rel_error = compute_metrics_per_step_samplewise(pred, ground_truth)
        #pointwise_rel_per = compute_pointwise_relative_per_step(pred, ground_truth)
        per_step_results[name] = {'mae': mae_per, 'rmse': rmse_per, 'rel_l2': rel_l2_per, 'rel_error': rel_error}
        horizon_results[name] = compute_metrics_at_horizons(pred, ground_truth, [1,5,10])

    # Create per-step table (csv)
    # Columns: step, model_mae, model_rmse, model_rel_l2, model_pointwise_rel for each model
    steps_idx = np.arange(1, steps+1)
    data = {'step': steps_idx}
    for name in model_names:
        if name in per_step_results:
            data[f'{name}_mae'] = per_step_results[name]['mae']
            data[f'{name}_rmse'] = per_step_results[name]['rmse']
            data[f'{name}_rel_l2'] = per_step_results[name]['rel_l2']
            data[f'{name}_rel_error'] = per_step_results[name]['rel_error']
    df_per_step = pd.DataFrame(data)
    per_step_csv = os.path.join(table_dir, 'rollout_metrics_per_step.csv')
    df_per_step.to_csv(per_step_csv, index=False)
    print(f"Saved per-step metrics to {per_step_csv}")

    # Create key-horizon summary table
    rows = []
    for name in model_names:
        if name not in horizon_results:
            continue
        hr = horizon_results[name]
        rows.append({
            'Model': name,
            'MAE@1': hr.get('mae@1', np.nan),
            'RMSE@1': hr.get('rmse@1', np.nan),
            'RelL2@1': hr.get('rel_l2@1', np.nan),
            'MAE@5': hr.get('mae@5', np.nan),
            'RMSE@5': hr.get('rmse@5', np.nan),
            'RelL2@5': hr.get('rel_l2@5', np.nan),
            'MAE@10': hr.get('mae@10', np.nan),
            'RMSE@10': hr.get('rmse@10', np.nan),
            'RelL2@10': hr.get('rel_l2@10', np.nan),
        })
    df_summary = pd.DataFrame(rows)
    summary_csv = os.path.join(table_dir, 'summary_metrics.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}")

    # Also save as markdown
    summary_md = os.path.join(table_dir, 'summary_metrics.md')
    with open(summary_md, 'w') as f:
        f.write(df_summary.to_markdown(index=False))
    print(f"Saved summary markdown to {summary_md}")

    # Print results
    print("\n=== Summary Metrics ===")
    print(df_summary.to_string(index=False))
    print("\n=== Per-step MAE (first 5 steps) ===")
    print(df_per_step.head().to_string(index=False))

    return df_summary, df_per_step


if __name__ == '__main__':
    evaluate_all()