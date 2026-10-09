import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_metrics_per_step

import matplotlib as mpl
from metrics import compute_metrics_per_step_ci

mpl.rcParams['font.family'] = 'Times New Roman'


def plot_rollout_curves(predictions, ground_truth, model_names, save_dir='./outputs/figures'):
    """
    Plot RMSE and MAE rollout curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    steps = ground_truth.shape[1]
    x = np.arange(1, steps+1)

    # Compute per-step metrics for each model
    colors = ['red', 'blue', 'green','purple','orange']
    linestyles = ['--', '-', '-', '-', '-']
    markers = [None, 'o', '^', 'D', None]
    #colors = ['blue', 'green','purple','orange']
    rmse_all = {}
    rmse_lower_all = {}
    rmse_upper_all = {}

    rel_error_all = {}
    rel_error_lower_all = {}
    rel_error_upper_all = {}
    rel_error_all = {}
    for name in model_names:
        pred = predictions.get(name)
        if pred is None:
            continue
        # rel_error_per, rmse_per = compute_metrics_per_step(pred, ground_truth)
        (rel_mean,rmse_mean,rel_lower,rel_upper,rmse_lower,rmse_upper,) = compute_metrics_per_step_ci(pred,ground_truth,)
        # rmse_all[name] = rmse_per
        # rel_error_all[name] = rel_error_per
        rmse_all[name] = rmse_mean
        rmse_lower_all[name] = rmse_lower
        rmse_upper_all[name] = rmse_upper

        rel_error_all[name] = rel_mean
        rel_error_lower_all[name] = rel_lower
        rel_error_upper_all[name] = rel_upper
    # ===== 在这里检查 95% CI 的实际宽度 =====
    for name in model_names:

        check_indices = [0, 10, 20, 49]

        for idx in check_indices:
            print(
                f"step={idx + 1:3d}, "
                f"mean={rmse_all[name][idx].item():.6f}, "
                f"lower={rmse_lower_all[name][idx].item():.6f}, "
                f"upper={rmse_upper_all[name][idx].item():.6f}, "
                f"CI width="
                f"{(rmse_upper_all[name][idx] - rmse_lower_all[name][idx]).item():.6f}"
            )

    # ===== 然后再开始画图 =====

    # Plot RMSE
    fig, ax = plt.subplots(figsize=(8,6))
    i=0
    for name in model_names:
        if name in rmse_all:
            plt.plot(x, rmse_all[name], label=name, linestyle=linestyles[i],marker=markers[i], linewidth=2.5, color=colors[i],markevery=10)
            plt.fill_between(x,rmse_lower_all[name],rmse_upper_all[name],color=colors[i],alpha=0.18,linewidth=0,)
            i = i+1
    plt.xlabel('Rollout steps',fontsize=20)
    plt.ylabel('RMSE',fontsize=20)
    # ===== 新增：主图 y 轴使用 log scale =====
    ax.set_yscale('log')

   #plt.title('RMSE vs Rollout Step')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=17)
    plt.tick_params(
        axis='both',        
        which='major',    
        labelsize=18       
    )
    # ===== 在大图内部添加一个小框 =====
    axins = inset_axes(
        ax,
        width="30%",  # 小图宽度
        height="30%",  # 小图高度
        loc='lower left',
        #bbox_to_anchor=(0.08, 0.25, 1, 1),  # 控制小图位置
        bbox_to_anchor=(0.35, 0.05, 0.9, 0.9),  # 控制小图位置
        bbox_transform=ax.transAxes
    )

    i = 0
    for name in model_names:
        if name in rmse_all:
            axins.plot(x, rmse_all[name], linewidth=2.0, color=colors[i],linestyle=linestyles[i],marker=markers[i])
            i += 1

    # 小图只显示前几步
    axins.set_xlim(1, 2)

    # y 轴范围根据前5步自动调整
    y_min = min(np.min(rmse_all[name][:2]) for name in model_names if name in rmse_all)
    y_max = max(np.max(rmse_all[name][:2]) for name in model_names if name in rmse_all)
    axins.set_ylim(y_min, y_max)

    axins.grid(True, alpha=0.3)
    axins.tick_params(axis='both', which='major', labelsize=16)

    # ===== 在大图中框出对应区域，并连到小图 =====
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.2)

    rmse_path = os.path.join(save_dir, 'rollout_rmse.pdf')
    plt.savefig(rmse_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSE curve to {rmse_path}")

    # Plot MAE
    i=0
    fig, ax = plt.subplots(figsize=(8,6))
    for name in model_names:
        if name in rel_error_all:
            plt.plot(x, rel_error_all[name], label=name, linestyle=linestyles[i],marker=markers[i], linewidth=2.5,color=colors[i],markevery=10)
            plt.fill_between(x,rel_error_lower_all[name],rel_error_upper_all[name],color=colors[i],alpha=0.18,linewidth=0,)
            i=i+1
    plt.xlabel('Rollout steps',fontsize=20)
    plt.ylabel('Relative Error',fontsize=20)

    # ===== 新增：主图 y 轴使用 log scale =====
    ax.set_yscale('log')

    #plt.title('MAE vs Rollout Step')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=17)
    plt.tick_params(
        axis='both',        
        which='major',      
        labelsize=18
    )
    # ===== 在大图内部添加一个小框 =====
    axins = inset_axes(
        ax,
        width="30%",  # 小图宽度
        height="30%",  # 小图高度
        loc='lower left',
        #bbox_to_anchor=(0.08, 0.25, 1, 1),  # 控制小图位置
        bbox_to_anchor=(0.35, 0.05, 0.9, 0.9),  # 控制小图位置
        bbox_transform=ax.transAxes
    )

    i = 0
    for name in model_names:
        # if name in rmse_all:
        if name in rel_error_all:
            axins.plot(x,  rel_error_all[name], linewidth=2.0, color=colors[i],linestyle=linestyles[i],marker=markers[i])
            i += 1

    # 小图只显示前几步
    axins.set_xlim(1, 2)

    # y 轴范围根据前5步自动调整
    y_min = min(np.min( rel_error_all[name][:2]) for name in model_names if name in  rel_error_all)
    y_max = max(np.max( rel_error_all[name][:2]) for name in model_names if name in  rel_error_all)
    axins.set_ylim(y_min, y_max)

    axins.grid(True, alpha=0.3)
    axins.tick_params(axis='both', which='major', labelsize=14)

    # ===== 在大图中框出对应区域，并连到小图 =====
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.2)

    rel_error_path = os.path.join(save_dir, 'rollout_rel_error.pdf')
    plt.savefig(rel_error_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Relative error curve to {rel_error_path}")


def plot_sample_visualizations(predictions, ground_truth, model_names,
                               sample_indices, steps_to_show=[1,5,10],
                               save_dir='./outputs/figures'):
    """
    Plot sample visualizations for selected samples and steps.
    Only channel 0 is shown.
    Creates one figure per sample per step, with subplots for ground truth and each model.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Ensure predictions include ground truth as a 'Ground Truth' entry
    all_data = {**predictions, 'Ground Truth': ground_truth}
    model_names_with_gt = ['Ground Truth'] + model_names
    for sample_idx in sample_indices:
        for step in steps_to_show:
            step_idx = step - 1
            fig, axes = plt.subplots(1, len(model_names_with_gt), figsize=(4*len(model_names_with_gt), 4))
            if len(model_names_with_gt) == 1:
                axes = [axes]
            for col, name in enumerate(model_names_with_gt):
                if name == 'Ground Truth':
                    img = ground_truth[sample_idx, step_idx, 0].cpu().numpy()
                else:
                    pred = predictions.get(name)
                    if pred is None:
                        continue
                    img = pred[sample_idx, step_idx, 0].cpu().numpy()
                ax = axes[col]
                im = ax.imshow(img, cmap='viridis', aspect='auto')
                ax.set_title(name)
                ax.axis('off')
                # Add colorbar for each subplot? We'll add one shared.
            # Add a single colorbar
            fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
            fig.suptitle(f'Sample {sample_idx}, Step {step}', fontsize=16)
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'case_{sample_idx}_step_{step}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_path}")


def main():
    prediction_dir = './outputs/predictions'
    model_names = ['BiCNN','U-Net', 'Plain CNN', 'FNO','PPNN-cs']
    #model_names = ['U-Net', 'Plain CNN', 'FNO','PPNN-cs']
    # Load ground truth
    gt_path = os.path.join(prediction_dir, 'ground_truth_rollout.pt')
    ground_truth = torch.load(gt_path)
    # Load predictions
    predictions = {}
    for name in model_names:
        path = os.path.join(prediction_dir, f'{name}_rollout.pt')
        if os.path.exists(path):
            predictions[name] = torch.load(path)
        else:
            print(f"Warning: {path} not found.")

    # Plot rollout curves
    plot_rollout_curves(predictions, ground_truth, model_names)

    # Select 3 representative samples (random seed 42)
    np.random.seed(42)
    n_samples = ground_truth.shape[0]
    sample_indices = np.random.choice(n_samples, size=min(3, n_samples), replace=False)
    print(f"Selected sample indices: {sample_indices}")

    # Plot sample visualizations
    plot_sample_visualizations(predictions, ground_truth, model_names,
                               sample_indices, steps_to_show=[1,5,10])


if __name__ == '__main__':
    main()