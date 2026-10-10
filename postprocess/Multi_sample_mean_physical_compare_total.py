import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
from scipy.interpolate import griddata
import pandas as pd

import matplotlib as mpl
from scipy.stats import t
# -----------------------------   不同物理信息进行测试  -----------------------#
mpl.rcParams['font.family'] = 'Times New Roman'

test_num = 7
n_turbine = 1


for step_num in range(1):
    file_path = 'postprocess/data/robust/multi_step_mean/one_turbine/%d/accumulated_u0_one_random.pt' % (
        test_num)
    file_path1 = 'postprocess/data/robust/multi_step_mean/one_turbine/%d/accumulated_u0_noise_one_random.pt' % (
        test_num)
    file_path_true = 'postprocess/data/robust/multi_step_mean/one_turbine/%d/true_u0_one_random.pt' % (
        test_num)
    file_partial_path = 'postprocess/data/robust/multi_step_mean/one_turbine/%d/accumulated_u0_one_partial_random.pt' % (
        test_num)
    point_path = 'points_XY_one.pt'



    # 定义 MSE 的损失函数
    mse_loss = nn.MSELoss(reduction='none')  # 不直接求均值，返回逐元素误差

    #############

    accumulated_u0 = torch.load(file_path)   #### our model
    accumulated_u0_data = torch.load(file_path1)  ### our model - no layer(I)
    true_u0 = torch.load(file_path_true)  ###### 真实值
    accumulated_u0_p = torch.load(file_partial_path)  # our partial model





    accumulated_u0_m=torch.sqrt(torch.sum(accumulated_u0[:,:,:]**2,dim=2,keepdim=True))
    accumulated_u0_data_m = torch.sqrt(torch.sum(accumulated_u0_data**2,dim=2,keepdim=True))
    true_u0_m = torch.sqrt(torch.sum(true_u0**2,dim=2,keepdim=True))
    print("true_u0_m.shape:",true_u0_m.shape)

    accumulated_u0_p_m = torch.sqrt(torch.sum(accumulated_u0_p ** 2, dim=2, keepdim=True))  # our partial model

    # =========================================================================================
    #                                   Relative Error
    # =========================================================================================

    # ----------------------------- our model ----------------------------- #
    mse_u0_pixel = torch.abs(accumulated_u0_m - true_u0_m)
    mse_true_pixel = torch.abs(true_u0_m - torch.zeros_like(true_u0_m))

    accumulated_u0_m_L2 = torch.mean(mse_u0_pixel / mse_true_pixel,dim=(3, 4))

    print("accumulated_u0_m_L2.shape:",accumulated_u0_m_L2.shape)

    # mean / max / min / std
    accumulated_u0_m_L2_mean = torch.mean(accumulated_u0_m_L2,dim=0)

    accumulated_u0_m_L2_max, _ = torch.max(accumulated_u0_m_L2,dim=0)

    accumulated_u0_m_L2_min, _ = torch.min(accumulated_u0_m_L2,dim=0)

    accumulated_u0_m_L2_std = torch.std(accumulated_u0_m_L2,dim=0)

    print("accumulated_u0_m_L2_mean.shape:",accumulated_u0_m_L2_mean.shape)

    # ----------------------------- 95% CI ----------------------------- #
    num_samples = accumulated_u0_m_L2.shape[0]

    # t distribution critical value
    t_value = t.ppf(0.975,df=num_samples - 1)

    # standard error
    accumulated_u0_m_L2_se = (accumulated_u0_m_L2_std / np.sqrt(num_samples))

    # 95% confidence interval half-width
    accumulated_u0_m_L2_ci = (t_value * accumulated_u0_m_L2_se)

    # =========================================================================================
    #                          our model w/o Layer(I)
    # =========================================================================================

    mse_u0_data_pixel = torch.abs(accumulated_u0_data_m - true_u0_m)

    accumulated_u0_data_m_L2 = torch.mean(mse_u0_data_pixel / mse_true_pixel, dim=(3, 4))

    accumulated_u0_data_m_L2_mean = torch.mean(accumulated_u0_data_m_L2,dim=0)

    accumulated_u0_data_m_L2_max, _ = torch.max(accumulated_u0_data_m_L2,dim=0)

    accumulated_u0_data_m_L2_min, _ = torch.min(accumulated_u0_data_m_L2,dim=0)

    accumulated_u0_data_m_L2_std = torch.std(accumulated_u0_data_m_L2,dim=0)

    # 95% confidence interval
    accumulated_u0_data_m_L2_se = (accumulated_u0_data_m_L2_std/ np.sqrt(accumulated_u0_data_m_L2.shape[0]))

    accumulated_u0_data_m_L2_ci = (
            t.ppf(
                0.975,
                df=accumulated_u0_data_m_L2.shape[0] - 1
            )
            * accumulated_u0_data_m_L2_se
    )

    print(
        "accumulated_u0_data_m_L2_mean.shape:",
        accumulated_u0_data_m_L2_mean.shape
    )



    # =========================================================================================
    #                                   New model
    # =========================================================================================
    mse_u0_p_pixel = torch.abs(
        accumulated_u0_p_m - true_u0_m
    )

    accumulated_u0_p_m_L2 = torch.mean(
        mse_u0_p_pixel / mse_true_pixel,
        dim=(3, 4)
    )

    # mean
    accumulated_u0_p_m_L2_mean = torch.mean(
        accumulated_u0_p_m_L2,
        dim=0
    )

    # max
    accumulated_u0_p_m_L2_max, _ = torch.max(
        accumulated_u0_p_m_L2,
        dim=0
    )

    # min
    accumulated_u0_p_m_L2_min, _ = torch.min(
        accumulated_u0_p_m_L2,
        dim=0
    )

    # std
    accumulated_u0_p_m_L2_std = torch.std(
        accumulated_u0_p_m_L2,
        dim=0
    )

    # ---------------------- 95% confidence interval ---------------------- #

    num_samples_p = accumulated_u0_p_m_L2.shape[0]

    t_value_p = t.ppf(
        0.975,
        df=num_samples_p - 1
    )

    accumulated_u0_p_m_L2_se = (
            accumulated_u0_p_m_L2_std
            / np.sqrt(num_samples_p)
    )

    accumulated_u0_p_m_L2_ci = (
            t_value_p * accumulated_u0_p_m_L2_se
    )

    # =========================================================================================
    #                        你原来的 nRMSE 部分，保持不变
    # =========================================================================================

    mse_u0_pixel_1 = torch.abs(
        accumulated_u0_m - true_u0_m
    )

    mse_true_pixel_1 = 8

    accumulated_u0_m_L2_1 = torch.mean(
        mse_u0_pixel_1 / mse_true_pixel_1,
        dim=(3, 4)
    )


    # =========================================================================================
    #                        Tensor -> numpy
    # =========================================================================================

    # ----------------------------- our model ----------------------------- #
    accumulated_u0_m_L2_mean = (
        accumulated_u0_m_L2_mean
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_m_L2_max = (
        accumulated_u0_m_L2_max
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_m_L2_min = (
        accumulated_u0_m_L2_min
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_m_L2_std = (
        accumulated_u0_m_L2_std
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_m_L2_ci = (
        accumulated_u0_m_L2_ci
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    # ----------------------------- New model ----------------------------- #

    accumulated_u0_p_m_L2_mean = (
        accumulated_u0_p_m_L2_mean
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_p_m_L2_max = (
        accumulated_u0_p_m_L2_max
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_p_m_L2_min = (
        accumulated_u0_p_m_L2_min
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_p_m_L2_std = (
        accumulated_u0_p_m_L2_std
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_p_m_L2_ci = (
        accumulated_u0_p_m_L2_ci
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    print(
        "accumulated_u0_m_L2_mean[0]:",
        accumulated_u0_m_L2_mean[0]
    )

    print(
        "accumulated_u0_m_L2_mean[4]:",
        accumulated_u0_m_L2_mean[4]
    )

    print(
        "accumulated_u0_m_L2_mean[9]:",
        accumulated_u0_m_L2_mean[9]
    )

    # ---------------------- our model w/o Layer(I) ---------------------- #
    accumulated_u0_data_m_L2_mean = (
        accumulated_u0_data_m_L2_mean
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_data_m_L2_max = (
        accumulated_u0_data_m_L2_max
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_data_m_L2_min = (
        accumulated_u0_data_m_L2_min
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_data_m_L2_std = (
        accumulated_u0_data_m_L2_std
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_data_m_L2_ci = (
        accumulated_u0_data_m_L2_ci
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )


    # =========================================================================================
    #                              Prediction steps
    # =========================================================================================

    if n_turbine == 1:
        N = 120

    steps = np.arange(1, N + 1)

    print(
        "accumulated_u0_m_L2_mean[N-1]:",
        accumulated_u0_m_L2_mean[N - 1]
    )



    print(
        "accumulated_u0_data_m_L2_mean[N-1]:",
        accumulated_u0_data_m_L2_mean[N - 1]
    )

    # =========================================================================================
    #                         Relative Error curve + 95% CI
    # =========================================================================================

    plt.figure(figsize=(8, 6))



    # ---------------------- our model w/o Layer(I) ---------------------- #
    plt.plot(
        steps,
        accumulated_u0_data_m_L2_mean[:N],
        linestyle='--',
        label="PPNN-cs w/o Layer(I)",
        color='blue',
        linewidth=2.5
    )

    plt.fill_between(
        steps,
        accumulated_u0_data_m_L2_mean[:N]
        - accumulated_u0_data_m_L2_ci[:N],

        accumulated_u0_data_m_L2_mean[:N]
        + accumulated_u0_data_m_L2_ci[:N],

        color='blue',
        alpha=0.18,
        linewidth=0
    )
    # ----------------------------- our model ----------------------------- #
    plt.plot(
        steps,
        accumulated_u0_m_L2_mean[:N],
        label="PPNN-cs",
        color='orange',
        linewidth=2.5
    )

    plt.fill_between(
        steps,
        accumulated_u0_m_L2_mean[:N]
        - accumulated_u0_m_L2_ci[:N],

        accumulated_u0_m_L2_mean[:N]
        + accumulated_u0_m_L2_ci[:N],

        color='orange',
        alpha=0.18,
        linewidth=0
    )
    # ----------------------------- New model ----------------------------- #

    plt.plot(
        steps,
        accumulated_u0_p_m_L2_mean[:N],
        linestyle= '-',
        marker='^',
        label=r'PPNN-cs $\mathcal{P}_1$',
        color='purple',
        linewidth=2.5,
        markevery = 10
    )

    plt.fill_between(
        steps,
        accumulated_u0_p_m_L2_mean[:N]
        - accumulated_u0_p_m_L2_ci[:N],

        accumulated_u0_p_m_L2_mean[:N]
        + accumulated_u0_p_m_L2_ci[:N],

        color='purple',
        alpha=0.18,
        linewidth=0
    )



    # ----------------------------- figure settings ----------------------------- #
    plt.xlabel(
        "Rollout steps",
        fontsize=20
    )

    plt.ylabel(
        "Relative Error",
        fontsize=20
    )

    plt.xticks(
        fontsize=18
    )

    plt.yticks(
        fontsize=18
    )

    plt.legend(
        fontsize=18
    )

    plt.grid()

    plt.show()

    # =========================================================================================
    #                                      RMSE
    # =========================================================================================

    # =========================================================================================
    #                                  our model
    # =========================================================================================

    rmse_per_sample_per_case = torch.sqrt(
        (
                (accumulated_u0_m - true_u0_m) ** 2
        ).mean(
            dim=(2, 3, 4)
        )
    )

    # mean RMSE
    rmse_per_case = torch.mean(
        rmse_per_sample_per_case,
        dim=0
    )

    # std
    rmse_std = torch.std(
        rmse_per_sample_per_case,
        dim=0
    )

    # standard error
    rmse_se = (
            rmse_std
            / np.sqrt(rmse_per_sample_per_case.shape[0])
    )

    # t critical value
    t_value_rmse = t.ppf(
        0.975,
        df=rmse_per_sample_per_case.shape[0] - 1
    )

    # 95% CI
    rmse_ci = (
            t_value_rmse * rmse_se
    )

    # to numpy
    rmse = (
        rmse_per_case
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    rmse_ci = (
        rmse_ci
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )


    # =========================================================================================
    #                           our model w/o Layer(I)
    # =========================================================================================

    rmse_per_sample_per_case_Data = torch.sqrt(
        (
                (accumulated_u0_data_m - true_u0_m) ** 2
        ).mean(
            dim=(2, 3, 4)
        )
    )

    rmse_per_case_Data = torch.mean(
        rmse_per_sample_per_case_Data,
        dim=0
    )

    rmse_std_Data = torch.std(
        rmse_per_sample_per_case_Data,
        dim=0
    )

    rmse_se_Data = (
            rmse_std_Data
            / np.sqrt(rmse_per_sample_per_case_Data.shape[0])
    )

    t_value_rmse_Data = t.ppf(
        0.975,
        df=rmse_per_sample_per_case_Data.shape[0] - 1
    )

    rmse_ci_Data = (
            t_value_rmse_Data
            * rmse_se_Data
    )

    rmse_Data = (
        rmse_per_case_Data
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    rmse_ci_Data = (
        rmse_ci_Data
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    # =========================================================================================
    #                                 New model RMSE
    # =========================================================================================

    rmse_per_sample_per_case_p = torch.sqrt(
        (
                (accumulated_u0_p_m - true_u0_m) ** 2
        ).mean(
            dim=(2, 3, 4)
        )
    )

    # mean
    rmse_per_case_p = torch.mean(
        rmse_per_sample_per_case_p,
        dim=0
    )

    # std
    rmse_std_p = torch.std(
        rmse_per_sample_per_case_p,
        dim=0
    )

    # standard error
    rmse_se_p = (
            rmse_std_p
            / np.sqrt(rmse_per_sample_per_case_p.shape[0])
    )

    # t critical value
    t_value_rmse_p = t.ppf(
        0.975,
        df=rmse_per_sample_per_case_p.shape[0] - 1
    )

    # 95% CI
    rmse_ci_p = (
            t_value_rmse_p * rmse_se_p
    )

    # Tensor -> numpy
    rmse_p = (
        rmse_per_case_p
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    rmse_ci_p = (
        rmse_ci_p
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    # =========================================================================================
    #                             RMSE curve + 95% CI
    # =========================================================================================

    plt.figure(figsize=(8, 6))



    # ---------------------- our model w/o Layer(I) ---------------------- #
    plt.plot(
        steps,
        rmse_Data[:N],
        linestyle= '--',
        label="PPNN-cs w/o Layer(I)",
        color='blue',
        linewidth=2.5

    )

    plt.fill_between(
        steps,
        rmse_Data[:N] - rmse_ci_Data[:N],
        rmse_Data[:N] + rmse_ci_Data[:N],
        color='blue',
        alpha=0.18,
        linewidth=0
    )
    # ----------------------------- our model ----------------------------- #
    plt.plot(
        steps,
        rmse[:N],
        label="PPNN-cs",
        color='orange',
        linewidth=2.5
    )

    plt.fill_between(
        steps,
        rmse[:N] - rmse_ci[:N],
        rmse[:N] + rmse_ci[:N],
        color='orange',
        alpha=0.18,
        linewidth=0
    )

    # ----------------------------- New model ----------------------------- #

    plt.plot(
        steps,
        rmse_p[:N],
        linestyle= '-',
        marker='^',
        label=r'PPNN-cs $\mathcal{P}_1$',
        color='purple',
        linewidth=2.5,
        markevery = 10
    )

    plt.fill_between(
        steps,
        rmse_p[:N] - rmse_ci_p[:N],
        rmse_p[:N] + rmse_ci_p[:N],
        color='purple',
        alpha=0.18,
        linewidth=0
    )


    # ----------------------------- figure settings ----------------------------- #
    plt.xlabel(
        "Rollout steps",
        fontsize=20
    )

    plt.ylabel(
        "RMSE",
        fontsize=20
    )

    plt.xticks(
        fontsize=18
    )

    plt.yticks(
        fontsize=18
    )

    plt.legend(
        fontsize=18
    )

    plt.grid()

    plt.show()
    #############################################   图像对比  ############################################################
    points_XY = torch.load(point_path, map_location='cpu')
    x = points_XY[0].cpu().numpy().flatten()  # x 坐标
    y = points_XY[1].cpu().numpy().flatten()  # y 坐标

    #####################################################################################################################
    #------------------------------------------ 统计指标 ----------------------------------------------------------------#
    ## 选定测试的步数
    statisc_steps = 80

    if n_turbine == 1:
        # our model
        steps_data_pre = accumulated_u0[:, statisc_steps - 1]


        # 部分物理信息
        steps_data_pre_partial = accumulated_u0_p[:, statisc_steps - 1]



        # our model w/o Layer(I)
        print("accumulated_u0_data.shape:", accumulated_u0_data.shape)
        steps_data_pre_data = accumulated_u0_data[:, statisc_steps - 1]


    steps_data_true = true_u0[:, statisc_steps - 1]
    print("steps_data_pre.shape:", steps_data_pre.shape)

    pred = torch.sqrt(torch.sum(steps_data_pre**2,dim=1,keepdim=True))
    true = torch.sqrt(torch.sum(steps_data_true**2,dim=1,keepdim=True))
    pred = pred[:,0,:,:]
    true = true[:,0,:,:]
    print("pred.shape:", pred.shape)

    if n_turbine == 1:
        pred_partial = torch.sqrt(
            torch.sum(steps_data_pre_partial ** 2, dim=1, keepdim=True)
        )

        pred_partial = pred_partial[:, 0, :, :]

        pred_data = torch.sqrt(
            torch.sum(steps_data_pre_data ** 2, dim=1, keepdim=True)
        )
        pred_data = pred_data[:, 0, :, :]

    # 定义 MSE 和 MAE 的损失函数
    mse_loss = nn.MSELoss(reduction='none')  # 不直接求均值，返回逐元素误差
    mae_loss = nn.L1Loss(reduction='none')   # 用于计算 MAE

    # 计算逐样本的误差
    mse_per_pixel = mse_loss(pred, true)  # 形状仍为 [N, 8, 10]
    mse_per_sample = torch.mean(mse_per_pixel, dim=(1, 2))  # 逐样本 MSE，形状 [N]
    rmse_per_sample = torch.sqrt(mse_per_sample)  # 逐样本 RMSE

    # =========================================================================================
    # Baseline: per-sample RMSE
    # =========================================================================================

    if n_turbine == 1:
        mse_per_pixel_partial = mse_loss(
            pred_partial,
            true
        )

        mse_per_sample_partial = torch.mean(
            mse_per_pixel_partial,
            dim=(1, 2)
        )

        rmse_per_sample_partial = torch.sqrt(
            mse_per_sample_partial
        )

        print(
            f"partial PDE 平均RMSE: "
            f"{torch.mean(rmse_per_sample_partial).item():.4f}\n"
            f"partial PDE RMSE方差: "
            f"{torch.var(rmse_per_sample_partial, unbiased=False).item():.8f}\n"
            f"partial PDE Baseline 最大RMSE: "
            f"{torch.max(rmse_per_sample_partial).item():.4f}\n"
            f"partial PDE Baseline 最小RMSE: "
            f"{torch.min(rmse_per_sample_partial).item():.4f}"
        )
    # our model w/o Layer(I)
    if n_turbine == 1:
        mse_per_pixel_data = mse_loss(pred_data, true)

        mse_per_sample_data = torch.mean(
            mse_per_pixel_data,
            dim=(1, 2)
        )

        rmse_per_sample_data = torch.sqrt(
            mse_per_sample_data
        )

        print(
            f"our model w/o Layer(I) 平均RMSE: "
            f"{torch.mean(rmse_per_sample_data).item():.4f}\n"
            f"方差: "
            f"{torch.var(rmse_per_sample_data, unbiased=False).item():.8f}\n"
            f"最大值: "
            f"{torch.max(rmse_per_sample_data).item():.4f}\n"
            f"最小值: "
            f"{torch.min(rmse_per_sample_data).item():.4f}"
        )


    # 计算逐样本的相对误差
    mse_true_pixel = mse_loss(true,torch.zeros_like(true))
    mse_rel_sample = torch.sqrt(torch.mean(mse_per_pixel/mse_true_pixel, dim=(1, 2)) ) # 逐样本 relative_L2，形状 [N]



    print(f"平均MSE: {torch.mean(mse_per_sample).item():.4f}\n"
        f"方差: {torch.var(mse_per_sample, unbiased=False).item():.8f}\n"
        f"最大值: {torch.max(mse_per_sample).item():.4f}\n"
        f"最小值: {torch.min(mse_per_sample).item():.4f}"
    )
    print(f"平均RMSE: {torch.mean(rmse_per_sample).item():.4f}\n"
        f"方差: {torch.var(rmse_per_sample, unbiased=False).item():.8f}\n"
        f"最大值: {torch.max(rmse_per_sample).item():.4f}\n"
        f"最小值: {torch.min(rmse_per_sample).item():.4f}"
          )



