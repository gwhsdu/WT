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
    file_path_Adam = 'postprocess/data/robust/multi_step_mean/one_turbine/%d_Adam/accumulated_u0_noise_one_Adam_random.pt' % (
        test_num)

    file_partial_path = 'postprocess/data/robust/multi_step_mean/one_turbine/%d/accumulated_u0_one_Adam_PDE_random.pt' % (
        test_num)
    point_path = 'points_XY_one.pt'

    ####################################################################################################################


    # 定义 MSE 的损失函数
    mse_loss = nn.MSELoss(reduction='none')  # 不直接求均值，返回逐元素误差

    #############

    accumulated_u0 = torch.load(file_path)   #### our model
    accumulated_u0_data = torch.load(file_path1)   ### our model
    true_u0 = torch.load(file_path_true)  ###### 真实值
    accumulated_u0_Adam = torch.load(file_path_Adam)   # Adam





    accumulated_u0_m=torch.sqrt(torch.sum(accumulated_u0[:,:,:]**2,dim=2,keepdim=True))
    accumulated_u0_data_m = torch.sqrt(torch.sum(accumulated_u0_data**2,dim=2,keepdim=True))
    true_u0_m = torch.sqrt(torch.sum(true_u0**2,dim=2,keepdim=True))
    print("true_u0_m.shape:",true_u0_m.shape)

    accumulated_u0_Adam_m=torch.sqrt(torch.sum(accumulated_u0_Adam[:,:,:]**2,dim=2,keepdim=True))


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

    t_value = t.ppf(0.975,df=num_samples - 1)

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
    #                                      Baseline
    # =========================================================================================

    print(
        "accumulated_u0_Adam_m.shape:",
        accumulated_u0_Adam_m.shape
    )

    print(
        "true_u0_m.shape:",
        true_u0_m.shape
    )

    mse_u0_Adam_pixel = torch.abs(
        accumulated_u0_Adam_m - true_u0_m
    )

    accumulated_u0_Adam_m_L2 = torch.mean(
        mse_u0_Adam_pixel / mse_true_pixel,
        dim=(3, 4)
    )

    accumulated_u0_Adam_m_L2_mean = torch.mean(
        accumulated_u0_Adam_m_L2,
        dim=0
    )

    accumulated_u0_Adam_m_L2_max, _ = torch.max(
        accumulated_u0_Adam_m_L2,
        dim=0
    )

    accumulated_u0_Adam_m_L2_min, _ = torch.min(
        accumulated_u0_Adam_m_L2,
        dim=0
    )

    accumulated_u0_Adam_m_L2_std = torch.std(
        accumulated_u0_Adam_m_L2,
        dim=0
    )

    # 95% confidence interval
    accumulated_u0_Adam_m_L2_se = (
            accumulated_u0_Adam_m_L2_std
            / np.sqrt(accumulated_u0_Adam_m_L2.shape[0])
    )

    accumulated_u0_Adam_m_L2_ci = (
            t.ppf(
                0.975,
                df=accumulated_u0_Adam_m_L2.shape[0] - 1
            )
            * accumulated_u0_Adam_m_L2_se
    )


    # =========================================================================================
    #                        原来的 nRMSE 部分，保持不变
    # =========================================================================================

    mse_u0_pixel_1 = torch.abs(
        accumulated_u0_m - true_u0_m
    )

    mse_true_pixel_1 = 8

    accumulated_u0_m_L2_1 = torch.mean(
        mse_u0_pixel_1 / mse_true_pixel_1,
        dim=(3, 4)
    )

    mse_u0_Adam_pixel_1 = torch.abs(
        accumulated_u0_Adam_m - true_u0_m
    )

    accumulated_u0_Adam_m_L2_1 = torch.mean(
        mse_u0_Adam_pixel_1 / mse_true_pixel_1,
        dim=(3, 4)
    )

    relative_error = (
            accumulated_u0_m_L2_1[:, 119]
            - accumulated_u0_Adam_m_L2_1[:, 119]
    )

    for kk in range(relative_error.shape[0]):
        print("kk:", kk)
        print("relative_error:", relative_error[kk])

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

    # ----------------------------- Baseline ----------------------------- #
    accumulated_u0_Adam_m_L2_mean = (
        accumulated_u0_Adam_m_L2_mean
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_Adam_m_L2_max = (
        accumulated_u0_Adam_m_L2_max
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_Adam_m_L2_min = (
        accumulated_u0_Adam_m_L2_min
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_Adam_m_L2_std = (
        accumulated_u0_Adam_m_L2_std
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    accumulated_u0_Adam_m_L2_ci = (
        accumulated_u0_Adam_m_L2_ci
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
        "accumulated_u0_Adam_m_L2_mean[N-1]:",
        accumulated_u0_Adam_m_L2_mean[N - 1]
    )

    print(
        "accumulated_u0_data_m_L2_mean[N-1]:",
        accumulated_u0_data_m_L2_mean[N - 1]
    )

    # =========================================================================================
    #                         Relative Error curve + 95% CI
    # =========================================================================================

    plt.figure(figsize=(8, 6))

    # ----------------------------- Baseline ----------------------------- #
    plt.plot(
        steps,
        accumulated_u0_Adam_m_L2_mean[:N],
        linestyle= '-',
        marker= 'o',
        label="Baseline",
        color='red',
        linewidth=2.5,
        markevery=10
    )

    plt.fill_between(
        steps,
        accumulated_u0_Adam_m_L2_mean[:N]
        - accumulated_u0_Adam_m_L2_ci[:N],

        accumulated_u0_Adam_m_L2_mean[:N]
        + accumulated_u0_Adam_m_L2_ci[:N],

        color='red',
        alpha=0.18,
        linewidth=0
    )

    # ---------------------- our model w/o Layer(I) ---------------------- #
    plt.plot(
        steps,
        accumulated_u0_data_m_L2_mean[:N],
        linestyle='--',
        label="PPNN-cs w/o Layer(I)",
        color='blue',
        linewidth=2.5,
        markevery=10
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
        linewidth=2.5,
        markevery=10
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
    #                                      Baseline
    # =========================================================================================

    rmse_per_sample_per_case_Adam = torch.sqrt(
        (
                (accumulated_u0_Adam_m - true_u0_m) ** 2
        ).mean(
            dim=(2, 3, 4)
        )
    )

    rmse_per_case_Adam = torch.mean(
        rmse_per_sample_per_case_Adam,
        dim=0
    )

    rmse_std_Adam = torch.std(
        rmse_per_sample_per_case_Adam,
        dim=0
    )

    rmse_se_Adam = (
            rmse_std_Adam
            / np.sqrt(rmse_per_sample_per_case_Adam.shape[0])
    )

    t_value_rmse_Adam = t.ppf(
        0.975,
        df=rmse_per_sample_per_case_Adam.shape[0] - 1
    )

    rmse_ci_Adam = (
            t_value_rmse_Adam
            * rmse_se_Adam
    )

    rmse_Adam = (
        rmse_per_case_Adam
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    rmse_ci_Adam = (
        rmse_ci_Adam
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
    #                             RMSE curve + 95% CI
    # =========================================================================================

    plt.figure(figsize=(8, 6))

    # ----------------------------- Baseline ----------------------------- #
    plt.plot(
        steps,
        rmse_Adam[:N],
        linestyle= '-',
        marker='o',
        label="Baseline",
        color='red',
        linewidth=2.5,
        markevery=10
    )

    plt.fill_between(
        steps,
        rmse_Adam[:N] - rmse_ci_Adam[:N],
        rmse_Adam[:N] + rmse_ci_Adam[:N],
        color='red',
        alpha=0.18,
        linewidth=0
    )

    # ---------------------- our model w/o Layer(I) ---------------------- #
    plt.plot(
        steps,
        rmse_Data[:N],
        linestyle='--',
        label="PPNN-cs w/o Layer(I)",
        color='blue',
        linewidth=2.5,
        markevery=10
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
        linewidth=2.5,
        markevery=10
    )

    plt.fill_between(
        steps,
        rmse[:N] - rmse_ci[:N],
        rmse[:N] + rmse_ci[:N],
        color='orange',
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
    print("")




    #####################################################################################################################

    print("accumulated_u0.shape:",accumulated_u0.shape)
    accumulated_u0 = torch.load(file_path)  #### our model-3D
    accumulated_u0_data = torch.load(file_path1)  ### our model-3D-data
    true_u0 = torch.load(file_path_true)  ###### 真实值
    accumulated_u0_Adam = torch.load(file_path_Adam)  # Adam
     #------ 选择的样本和预测步数
    sample_num = 50
    pre_step_1 = 10
    pre_step_2 = 30
    pre_step_3 = 60
    pre_step_end = 80
    pre_step_s = 120
    ## 真实值
    true_u0_select_1 = true_u0[sample_num,pre_step_1-1]
    true_u0_select_2 = true_u0[sample_num,pre_step_2-1]
    true_u0_select_3 = true_u0[sample_num,pre_step_3-1]
    true_u0_select_end = true_u0[sample_num,pre_step_end-1]
    true_u0_select_s = true_u0[sample_num, pre_step_s - 1]
    ## our model
    if n_turbine==1 or n_turbine==3:
        predict_cs_select_1 = accumulated_u0[sample_num,pre_step_1-1]
        predict_cs_select_2 = accumulated_u0[sample_num, pre_step_2-1]
        predict_cs_select_3 = accumulated_u0[sample_num, pre_step_3-1]
        predict_cs_select_end = accumulated_u0[sample_num, pre_step_end-1]
        predict_cs_select_s = accumulated_u0[sample_num, pre_step_s - 1]
    ## Black box
    predict_black_select_1 = accumulated_u0_Adam[sample_num, pre_step_1-1]
    predict_black_select_2 = accumulated_u0_Adam[sample_num, pre_step_2-1]
    predict_black_select_3 = accumulated_u0_Adam[sample_num, pre_step_3-1]
    predict_black_select_end = accumulated_u0_Adam[sample_num, pre_step_end-1]
    predict_black_select_s = accumulated_u0_Adam[sample_num, pre_step_s - 1]

    ##-------------------- 总的风速
    ## 真实值
    true_uo_t_1 = torch.sqrt(torch.sum(true_u0_select_1**2,dim=0,keepdim=True))
    true_uo_t_2 = torch.sqrt(torch.sum(true_u0_select_2 ** 2, dim=0, keepdim=True))
    true_uo_t_3 = torch.sqrt(torch.sum(true_u0_select_3 ** 2, dim=0, keepdim=True))
    true_uo_t_end = torch.sqrt(torch.sum(true_u0_select_end ** 2, dim=0, keepdim=True))
    true_uo_t_s = torch.sqrt(torch.sum(true_u0_select_s ** 2, dim=0, keepdim=True))

    ## PPNN_cs
    predict_cs_t_1 = torch.sqrt(torch.sum(predict_cs_select_1**2,dim=0,keepdim=True))
    predict_cs_t_2 = torch.sqrt(torch.sum(predict_cs_select_2**2,dim=0,keepdim=True))
    predict_cs_t_3 = torch.sqrt(torch.sum(predict_cs_select_3**2,dim=0,keepdim=True))
    predict_cs_t_end = torch.sqrt(torch.sum(predict_cs_select_end**2,dim=0,keepdim=True))
    predict_cs_t_s = torch.sqrt(torch.sum(predict_cs_select_s ** 2, dim=0, keepdim=True))

    ## Black box
    predict_black_t_1 = torch.sqrt(torch.sum(predict_black_select_1**2,dim=0,keepdim=True))
    predict_black_t_2 = torch.sqrt(torch.sum(predict_black_select_2**2,dim=0,keepdim=True))
    predict_black_t_3 = torch.sqrt(torch.sum(predict_black_select_3**2,dim=0,keepdim=True))
    predict_black_t_end = torch.sqrt(torch.sum(predict_black_select_end**2,dim=0,keepdim=True))
    predict_black_t_s = torch.sqrt(torch.sum(predict_black_select_s ** 2, dim=0, keepdim=True))

    ## 转换成numpy
    ## 真实值
    true_uo_n_1 = true_uo_t_1.cpu().numpy().flatten()
    true_uo_n_2 = true_uo_t_2.cpu().numpy().flatten()
    true_uo_n_3 = true_uo_t_3.cpu().numpy().flatten()
    true_uo_n_end = true_uo_t_end.cpu().numpy().flatten()
    true_uo_n_s = true_uo_t_s.cpu().numpy().flatten()

    ## PPNN_cs
    predict_cs_n_1 = predict_cs_t_1.cpu().numpy().flatten()
    predict_cs_n_2 = predict_cs_t_2.cpu().numpy().flatten()
    predict_cs_n_3 = predict_cs_t_3.cpu().numpy().flatten()
    predict_cs_n_end = predict_cs_t_end.cpu().numpy().flatten()
    predict_cs_n_s = predict_cs_t_s.cpu().numpy().flatten()

    ## Black box
    predict_black_n_1 = predict_black_t_1.cpu().numpy().flatten()
    predict_black_n_2 = predict_black_t_2.cpu().numpy().flatten()
    predict_black_n_3 = predict_black_t_3.cpu().numpy().flatten()
    predict_black_n_end = predict_black_t_end.cpu().numpy().flatten()
    predict_black_n_s = predict_black_t_s.cpu().numpy().flatten()

    # 创建网格
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

    # 插值生成平滑的 u 值
    grid_u_true_1 = griddata((x, y), true_uo_n_1, (grid_x, grid_y), method='cubic')
    grid_u_cs_1 = griddata((x, y), predict_cs_n_1, (grid_x, grid_y), method='cubic')
    grid_u_black_1 = griddata((x, y), predict_black_n_1, (grid_x, grid_y), method='cubic')
    grid_u_cs_error_1 = griddata((x, y), np.abs(predict_cs_n_1 - true_uo_n_1)/true_uo_n_1, (grid_x, grid_y), method='cubic')
    grid_u_black_error_1 = griddata((x, y), np.abs(predict_black_n_1 - true_uo_n_1) /true_uo_n_1, (grid_x, grid_y),method='cubic')

    grid_u_true_2 = griddata((x, y), true_uo_n_2, (grid_x, grid_y), method='cubic')
    grid_u_cs_2 = griddata((x, y), predict_cs_n_2, (grid_x, grid_y), method='cubic')
    grid_u_black_2 = griddata((x, y), predict_black_n_2, (grid_x, grid_y), method='cubic')
    grid_u_cs_error_2 = griddata((x, y), np.abs(predict_cs_n_2 - true_uo_n_2) / true_uo_n_2, (grid_x, grid_y), method='cubic')
    grid_u_black_error_2 = griddata((x, y), np.abs(predict_black_n_2 - true_uo_n_2) / true_uo_n_2, (grid_x, grid_y), method='cubic')

    grid_u_true_3 = griddata((x, y), true_uo_n_3, (grid_x, grid_y), method='cubic')
    grid_u_cs_3 = griddata((x, y), predict_cs_n_3, (grid_x, grid_y), method='cubic')
    grid_u_black_3 = griddata((x, y), predict_black_n_3, (grid_x, grid_y), method='cubic')
    grid_u_cs_error_3 = griddata((x, y), np.abs(predict_cs_n_3 - true_uo_n_3) / true_uo_n_3, (grid_x, grid_y), method='cubic')
    grid_u_black_error_3 = griddata((x, y), np.abs(predict_black_n_3 - true_uo_n_3) / true_uo_n_3, (grid_x, grid_y),method='cubic')

    grid_u_true_end = griddata((x, y), true_uo_n_end, (grid_x, grid_y), method='cubic')
    grid_u_cs_end = griddata((x, y), predict_cs_n_end, (grid_x, grid_y), method='cubic')
    grid_u_black_end = griddata((x, y), predict_black_n_end, (grid_x, grid_y), method='cubic')
    grid_u_cs_error_end = griddata((x, y), np.abs(predict_cs_n_end - true_uo_n_end) / true_uo_n_end, (grid_x, grid_y),method='cubic')
    grid_u_black_error_end = griddata((x, y), np.abs(predict_black_n_end - true_uo_n_end) / true_uo_n_end, (grid_x, grid_y), method='cubic')




    # 找出两个插值数据的最小值和最大值
    vmin = min(np.nanmin(grid_u_true_1), np.nanmin(grid_u_cs_1), np.nanmin(grid_u_true_2), np.nanmin(grid_u_cs_2), np.nanmin(grid_u_true_3),np.nanmin(grid_u_cs_3),np.nanmin(grid_u_true_end),np.nanmin(grid_u_cs_end))
    vmax = max(np.nanmax(grid_u_true_1), np.nanmax(grid_u_cs_1), np.nanmax(grid_u_true_2), np.nanmax(grid_u_cs_2), np.nanmax(grid_u_true_3),np.nanmax(grid_u_cs_3), np.nanmax(grid_u_true_end),np.nanmax(grid_u_cs_end))

    # ############# 画图(真实值和预测值同时显示)
    # 计算坐标范围，确定纵横比
    aspect_ratio = (max(x) - min(x)) / (max(y) - min(y))
    fig, axes = plt.subplots(4, 3, figsize=(8 * aspect_ratio, 8))
    # 调整子图水平间距和底部边距
    plt.subplots_adjust(wspace=0.1, bottom=0.25)
    # 定义每行要显示的行标题
    row_labels = [f"T+{pre_step_1}", f"T+{pre_step_2}", f"T+{pre_step_3}", f"T+{pre_step_end}"]

    # ----------------------  绘制第一行  --------------------------#
    # 绘制第一个子图
    im1 = axes[0, 0].imshow(grid_u_cs_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                            aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 0].tick_params(axis='both', labelsize=20)  # 字号设为12
    if n_turbine == 1:
        axes[0, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[0, 0].tick_params(axis='x', labelbottom=False, bottom=False)
    axes[0, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)
    axes[0, 0].set_title('PPNN-cs', fontsize=22)

    # 绘制第二个子图
    im2 = axes[0, 1].imshow(grid_u_true_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                            aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[0, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
    axes[0, 1].tick_params(axis='y', labelleft=False, left=False)
    axes[0, 1].set_title('LES ', fontsize=22)

    # 绘制第三个子图
    im3 = axes[0, 2].imshow(grid_u_black_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                            aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[0, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    elif n_turbine == 3:
        axes[0, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
        axes[0, 2].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
        axes[0, 2].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 2].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
    axes[0, 2].tick_params(axis='y', labelleft=False, left=False)
    axes[0, 2].set_title(' Baseline ', fontsize=22)

    # ----------------------  绘制第二行  --------------------------#
    im2_1 = axes[1, 0].imshow(grid_u_cs_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    axes[1, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

    if n_turbine == 1:
        axes[1, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[1, 0].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12

    axes[1, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)

    # 绘制第二个子图
    im2_2 = axes[1, 1].imshow(grid_u_true_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[1, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
    axes[1, 1].tick_params(axis='y', labelleft=False, left=False)

    # 绘制第三个子图
    im2_3 = axes[1, 2].imshow(grid_u_black_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[1, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[1, 2].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
    axes[1, 2].tick_params(axis='y', labelleft=False, left=False)

    # ----------------------  绘制第三行  --------------------------#
    # 绘制第一个子图
    im3_1 = axes[2, 0].imshow(grid_u_cs_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    axes[2, 0].tick_params(axis='both', labelsize=20)  # 字号设为12
    # # ----------------------------------------------#
    if n_turbine == 1:
        axes[2, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)


    axes[2, 0].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[2, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


    # 绘制第二个子图

    im3_2 = axes[2, 1].imshow(grid_u_true_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[2, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[2, 1].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[2, 1].tick_params(axis='y', labelleft=False, left=False)


    # 绘制第三个子图
    im3_3 = axes[2, 2].imshow(grid_u_black_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[2, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[2, 2].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[2, 2].tick_params(axis='y', labelleft=False, left=False)

    # ----------------------  绘制第四行  --------------------------#
    # 绘制第一个子图
    im3_1 = axes[3, 0].imshow(grid_u_cs_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    axes[3, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

    # # ----------------------------------------------#
    if n_turbine == 1:
        axes[3, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[3, 0].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[3, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


    # 绘制第二个子图
    im3_2 = axes[3, 1].imshow(grid_u_true_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower',
                              cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[3, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[3, 1].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[3, 1].tick_params(axis='y', labelleft=False, left=False)

    # 绘制第三个子图
    im3_3 = axes[3, 2].imshow(grid_u_black_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower',
                              cmap='coolwarm',
                              aspect='auto', vmin=vmin, vmax=vmax)
    if n_turbine == 1:
        axes[3, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)

    axes[3, 2].tick_params(axis='x', labelsize=20,labelbottom=False, bottom=False)  # 字号设为12
    axes[3, 2].tick_params(axis='y', labelleft=False, left=False)


    # # 获取子图位置信息
    #------------------------- 调整布局 ---------------------#
    plt.tight_layout(rect=[0.06, 0.12, 1, 0.98])
    #------------------------- 共享颜色条 --------------------#
    # 获取子图位置信息
    pos1 = axes[0, 0].get_position().bounds
    pos2 = axes[0, 1].get_position().bounds
    pos3 = axes[0, 2].get_position().bounds
    # ----- 前两个子图的共享颜色条 ----
    # 计算颜色条的左边界和宽度，使其横跨两个子图
    cax_left = pos1[0]
    cax_width = pos3[0] + pos3[2] - pos1[0]

    # 设置颜色条的位置参数（距离底部0.1，高度0.03）
    cax = fig.add_axes([cax_left, 0.09, cax_width, 0.03])

    # 创建水平颜色条
    cbar = fig.colorbar(im1, cax=cax, orientation='horizontal', label='u(m/s)')
    cbar.set_label('u(m/s)', fontsize=22)  # 设置颜色条标签
    cbar.ax.tick_params(labelsize=20)  # 刻度标签字号14

    #------------------------- 遍历：给每一行添加标注 ------------------------------#
    for i, label in enumerate(row_labels):
        fig.text(
            x=0.035,
            y=axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2,
            s=label,
            fontsize=22,
            # fontweight='bold',
            ha='center',
            va='center',
            rotation=0,
            color='black'
        )


    plt.show()


    ####################################################################################################################
    # -----------------------------------------   误差图   --------------------------------------------------------------#
    ####################################################################################################################

    # ================================================================================================================
    # 统一误差色标范围
    # ================================================================================================================
    vmin_error = 0.0
    vmax_error = 0.8

    # 计算坐标范围，确定纵横比
    aspect_ratio = (max(x) - min(x)) / (max(y) - min(y))

    # ================================================================================================================
    # 创建 4 × 2 子图
    #
    # 第一列：our model error
    # 第二列：Baseline error
    # ================================================================================================================
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(6 * aspect_ratio, 8)
    )

    # 调整子图间距，并给下面的共享 colorbar 留空间
    plt.subplots_adjust(
        wspace=0.10,
        bottom=0.20
    )

    # 每一行对应的 rollout step
    row_labels = [
        f"T+{pre_step_1}",
        f"T+{pre_step_2}",
        f"T+{pre_step_3}",
        f"T+{pre_step_end}"
    ]

    # ================================================================================================================
    #                                  第一行
    # ================================================================================================================

    # -------------------------------- our model -------------------------------- #
    im1 = axes[0, 0].imshow(
        grid_u_cs_error_1.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    # turbine position
    if n_turbine == 1:
        axes[0, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[0, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[0, 0].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[0, 0].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[0, 0].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[0, 0].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    axes[0, 0].set_title(
        'PPNN-cs',
        fontsize=22
    )

    # -------------------------------- Baseline -------------------------------- #
    im2 = axes[0, 1].imshow(
        grid_u_black_error_1.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[0, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[0, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[0, 1].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[0, 1].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[0, 1].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[0, 1].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    axes[0, 1].set_title(
        'Baseline',
        fontsize=22
    )

    # ================================================================================================================
    #                                  第二行
    # ================================================================================================================

    # -------------------------------- our model -------------------------------- #
    im3 = axes[1, 0].imshow(
        grid_u_cs_error_2.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[1, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[1, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[1, 0].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[1, 0].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[1, 0].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[1, 0].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # -------------------------------- Baseline -------------------------------- #
    im4 = axes[1, 1].imshow(
        grid_u_black_error_2.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[1, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[1, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[1, 1].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[1, 1].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[1, 1].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[1, 1].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # ================================================================================================================
    #                                  第三行
    # ================================================================================================================

    # -------------------------------- our model -------------------------------- #
    im5 = axes[2, 0].imshow(
        grid_u_cs_error_3.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[2, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[2, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[2, 0].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[2, 0].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[2, 0].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[2, 0].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # -------------------------------- Baseline -------------------------------- #
    im6 = axes[2, 1].imshow(
        grid_u_black_error_3.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[2, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[2, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[2, 1].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[2, 1].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[2, 1].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[2, 1].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # ================================================================================================================
    #                                  第四行
    # ================================================================================================================

    # -------------------------------- our model -------------------------------- #
    im7 = axes[3, 0].imshow(
        grid_u_cs_error_end.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[3, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[3, 0].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[3, 0].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[3, 0].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[3, 0].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[3, 0].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # -------------------------------- Baseline -------------------------------- #
    im8 = axes[3, 1].imshow(
        grid_u_black_error_end.T,
        extent=(min(x), max(x), min(y), max(y)),
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        vmin=vmin_error,
        vmax=vmax_error
    )

    if n_turbine == 1:
        axes[3, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    elif n_turbine == 3:
        axes[3, 1].plot(
            [500, 500],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[3, 1].plot(
            [1100, 1100],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

        axes[3, 1].plot(
            [1700, 1700],
            [600 - 50, 600 + 50],
            color='black',
            linewidth=2
        )

    axes[3, 1].tick_params(
        axis='x',
        labelbottom=False,
        bottom=False
    )

    axes[3, 1].tick_params(
        axis='y',
        labelleft=False,
        left=False
    )

    # ================================================================================================================
    #                                         调整布局
    # ================================================================================================================

    plt.tight_layout(
        rect=[0.07, 0.13, 1, 0.98]
    )

    # ================================================================================================================
    #                              两列误差图共享一个 colorbar
    # ================================================================================================================

    # 获取左右两个子图的位置
    pos_left = axes[0, 0].get_position().bounds
    pos_right = axes[0, 1].get_position().bounds

    # colorbar 从左边第一列开始，一直到右边第二列结束
    cax_left = pos_left[0]

    cax_width = (
            pos_right[0]
            + pos_right[2]
            - pos_left[0]
    )

    # colorbar 位置：
    # [left, bottom, width, height]
    cax = fig.add_axes([
        cax_left,
        0.085,
        cax_width,
        0.03
    ])

    # 所有误差图都是 vmin=0, vmax=2，
    # 所以任意一个 im 都可以作为共享 colorbar 的 mappable
    cbar = fig.colorbar(
        im1,
        cax=cax,
        orientation='horizontal'
    )

    # 推荐写 Relative Error，因为你这里画的是 |pred-true| / true
    cbar.set_label(
        'Relative Error',
        fontsize=22
    )

    cbar.ax.tick_params(
        labelsize=18
    )

    # 如果希望明确显示 0~2 的刻度
    # cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])

    # ================================================================================================================
    #                              每一行左侧添加 rollout step
    # ================================================================================================================

    for i, label in enumerate(row_labels):
        fig.text(
            x=0.035,

            y=(
                    axes[i, 0].get_position().y0
                    + axes[i, 0].get_position().height / 2
            ),

            s=label,

            fontsize=22,

            ha='center',
            va='center',

            rotation=0,

            color='black'
        )

    plt.show()

