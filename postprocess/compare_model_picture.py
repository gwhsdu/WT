import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
from scipy.interpolate import griddata
import pandas as pd

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
eps = 1e-8
####################################################################################################################
n_turbine = 1 # 1 or 3
type = 'no_inflow'  # 'no_inflow' or 'inflow'
#----------------------------------------------- 导入数据 -----------------------------------------------------------#
if n_turbine ==1:
    point_path = '/mnt/e/code/2025/predict/test_3d/points_XY_one.pt'
    if type == 'no_inflow':
        true_path = 'postprocess/data/compare_data/compare_no_inflow/one_turbine/ground_truth_rollout.pt'
        our_model_pre_path = 'postprocess/data/compare_data/compare_no_inflow/one_turbine/PPNN-cs_rollout.pt'
        other_model_pre_path = 'postprocess/data/compare_data/compare_no_inflow/one_turbine/FNO_rollout.pt'
    else:
        true_path = 'postprocess/data/compare_data/compare_inflow/one_turbine/ground_truth_rollout.pt'
        our_model_pre_path = 'postprocess/data/compare_data/compare_inflow/one_turbine/PPNN-cs_rollout.pt'
        other_model_pre_path = 'postprocess/data/compare_data/compare_inflow/one_turbine/CNN With Inflow_rollout.pt'
elif n_turbine ==3:
    point_path = '/mnt/e/code/2025/predict/test_3d/points_XY_three.pt'
    if type == 'no_inflow':
        true_path = 'postprocess/data/compare_data/compare_no_inflow/three_turbine/ground_truth_rollout.pt'
        our_model_pre_path = 'postprocess/data/compare_data/compare_no_inflow/three_turbine/PPNN-cs_rollout.pt'
        other_model_pre_path = 'postprocess/data/compare_data/compare_no_inflow/three_turbine/Plain CNN_rollout.pt'
    else:
        true_path = 'postprocess/data/compare_data/compare_inflow/three_turbine/ground_truth_rollout.pt'
        our_model_pre_path = 'postprocess/data/compare_data/compare_inflow/three_turbine/PPNN-cs_rollout.pt'
        other_model_pre_path = 'postprocess/data/compare_data/compare_inflow/three_turbine/CNN With Inflow_rollout.pt'

#-------------------------------------------------------------------------------------------------------------------#
accumulated_u0 = torch.load(our_model_pre_path)  #### PPNN-cs
true_u0 = torch.load(true_path)  ###### 真实值
accumulated_u0_Adam = torch.load(other_model_pre_path)  # 其他model

#———————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
#############################################   图像对比  ############################################################
points_XY = torch.load(point_path, map_location='cpu')
x = points_XY[0].cpu().numpy().flatten()  # x 坐标
y = points_XY[1].cpu().numpy().flatten()  # y 坐标

# ------ 选择的样本和预测步数
sample_num = 15
pre_step_1 = 10
pre_step_2 = 30
pre_step_3 = 60
pre_step_end = 80
pre_step_s = 100
## 真实值
true_u0_select_1 = true_u0[sample_num, pre_step_1 - 1]
true_u0_select_2 = true_u0[sample_num, pre_step_2 - 1]
true_u0_select_3 = true_u0[sample_num, pre_step_3 - 1]
true_u0_select_end = true_u0[sample_num, pre_step_end - 1]
true_u0_select_s = true_u0[sample_num, pre_step_s - 1]
## PPNN-cs
if n_turbine == 1 or n_turbine == 3:
    predict_cs_select_1 = accumulated_u0[sample_num, pre_step_1 - 1]
    predict_cs_select_2 = accumulated_u0[sample_num, pre_step_2 - 1]
    predict_cs_select_3 = accumulated_u0[sample_num, pre_step_3 - 1]
    predict_cs_select_end = accumulated_u0[sample_num, pre_step_end - 1]
    predict_cs_select_s = accumulated_u0[sample_num, pre_step_s - 1]
## Black box
predict_black_select_1 = accumulated_u0_Adam[sample_num, pre_step_1 - 1]
predict_black_select_2 = accumulated_u0_Adam[sample_num, pre_step_2 - 1]
predict_black_select_3 = accumulated_u0_Adam[sample_num, pre_step_3 - 1]
predict_black_select_end = accumulated_u0_Adam[sample_num, pre_step_end - 1]
predict_black_select_s = accumulated_u0_Adam[sample_num, pre_step_s - 1]

##-------------------- 总的风速
## 真实值
true_uo_t_1 = torch.sqrt(torch.sum(true_u0_select_1 ** 2, dim=0, keepdim=True))
true_uo_t_2 = torch.sqrt(torch.sum(true_u0_select_2 ** 2, dim=0, keepdim=True))
true_uo_t_3 = torch.sqrt(torch.sum(true_u0_select_3 ** 2, dim=0, keepdim=True))
true_uo_t_end = torch.sqrt(torch.sum(true_u0_select_end ** 2, dim=0, keepdim=True))
true_uo_t_s = torch.sqrt(torch.sum(true_u0_select_s ** 2, dim=0, keepdim=True))

## PPNN_cs
predict_cs_t_1 = torch.sqrt(torch.sum(predict_cs_select_1 ** 2, dim=0, keepdim=True))
predict_cs_t_2 = torch.sqrt(torch.sum(predict_cs_select_2 ** 2, dim=0, keepdim=True))
predict_cs_t_3 = torch.sqrt(torch.sum(predict_cs_select_3 ** 2, dim=0, keepdim=True))
predict_cs_t_end = torch.sqrt(torch.sum(predict_cs_select_end ** 2, dim=0, keepdim=True))
predict_cs_t_s = torch.sqrt(torch.sum(predict_cs_select_s ** 2, dim=0, keepdim=True))

## Black box
predict_black_t_1 = torch.sqrt(torch.sum(predict_black_select_1 ** 2, dim=0, keepdim=True))
predict_black_t_2 = torch.sqrt(torch.sum(predict_black_select_2 ** 2, dim=0, keepdim=True))
predict_black_t_3 = torch.sqrt(torch.sum(predict_black_select_3 ** 2, dim=0, keepdim=True))
predict_black_t_end = torch.sqrt(torch.sum(predict_black_select_end ** 2, dim=0, keepdim=True))
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


grid_u_cs_error_1 = griddata(
    (x, y),
    np.abs(predict_cs_n_1 - true_uo_n_1) / (np.abs(true_uo_n_1) + eps),
    (grid_x, grid_y),
    method='cubic'
)

grid_u_black_error_1 = griddata(
    (x, y),
    np.abs(predict_black_n_1 - true_uo_n_1) / (np.abs(true_uo_n_1) + eps),
    (grid_x, grid_y),
    method='cubic'
)

grid_u_true_2 = griddata((x, y), true_uo_n_2, (grid_x, grid_y), method='cubic')
grid_u_cs_2 = griddata((x, y), predict_cs_n_2, (grid_x, grid_y), method='cubic')
grid_u_black_2 = griddata((x, y), predict_black_n_2, (grid_x, grid_y), method='cubic')
grid_u_cs_error_2 = griddata(
    (x, y),
    np.abs(predict_cs_n_2 - true_uo_n_2) / (np.abs(true_uo_n_2) + eps),
    (grid_x, grid_y),
    method='cubic'
)

grid_u_black_error_2 = griddata(
    (x, y),
    np.abs(predict_black_n_2 - true_uo_n_2) / (np.abs(true_uo_n_2) + eps),
    (grid_x, grid_y),
    method='cubic'
)


grid_u_true_3 = griddata((x, y), true_uo_n_3, (grid_x, grid_y), method='cubic')
grid_u_cs_3 = griddata((x, y), predict_cs_n_3, (grid_x, grid_y), method='cubic')
grid_u_black_3 = griddata((x, y), predict_black_n_3, (grid_x, grid_y), method='cubic')

grid_u_cs_error_3 = griddata(
    (x, y),
    np.abs(predict_cs_n_3 - true_uo_n_3) / (np.abs(true_uo_n_3) + eps),
    (grid_x, grid_y),
    method='cubic'
)

grid_u_black_error_3 = griddata(
    (x, y),
    np.abs(predict_black_n_3 - true_uo_n_3) / (np.abs(true_uo_n_3) + eps),
    (grid_x, grid_y),
    method='cubic'
)


grid_u_true_end = griddata((x, y), true_uo_n_end, (grid_x, grid_y), method='cubic')
grid_u_cs_end = griddata((x, y), predict_cs_n_end, (grid_x, grid_y), method='cubic')
grid_u_black_end = griddata((x, y), predict_black_n_end, (grid_x, grid_y), method='cubic')

grid_u_cs_error_end = griddata(
    (x, y),
    np.abs(predict_cs_n_end - true_uo_n_end) / (np.abs(true_uo_n_end) + eps),
    (grid_x, grid_y),
    method='cubic'
)

grid_u_black_error_end = griddata(
    (x, y),
    np.abs(predict_black_n_end - true_uo_n_end) / (np.abs(true_uo_n_end) + eps),
    (grid_x, grid_y),
    method='cubic'
)



# 找出两个插值数据的最小值和最大值
vmin = min(np.nanmin(grid_u_true_1), np.nanmin(grid_u_cs_1), np.nanmin(grid_u_true_2), np.nanmin(grid_u_cs_2),
           np.nanmin(grid_u_true_3), np.nanmin(grid_u_cs_3), np.nanmin(grid_u_true_end), np.nanmin(grid_u_cs_end))
vmax = max(np.nanmax(grid_u_true_1), np.nanmax(grid_u_cs_1), np.nanmax(grid_u_true_2), np.nanmax(grid_u_cs_2),
           np.nanmax(grid_u_true_3), np.nanmax(grid_u_cs_3), np.nanmax(grid_u_true_end), np.nanmax(grid_u_cs_end))
if n_turbine == 3:
    vmin = 1.3 * vmin
# vmax = 0.98*vmax
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
if type == 'inflow':
    axes[0, 0].set_xlim(200, 700)
axes[0, 0].tick_params(axis='both', labelsize=20)  # 字号设为12
if n_turbine == 1:
    axes[0, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[0, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)

axes[0, 0].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[0, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)
axes[0, 0].set_title('PPNN-cs', fontsize=22)

# 绘制第二个子图
im2 = axes[0, 1].imshow(grid_u_true_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                        aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[0, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[0, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[0, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[0, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[0, 1].tick_params(axis='y', labelleft=False, left=False)
axes[0, 1].set_title('LES ', fontsize=22)

# 绘制第三个子图
im3 = axes[0, 2].imshow(grid_u_black_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                        aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[0, 2].set_xlim(200, 700)
if n_turbine == 1:
    axes[0, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[0, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 2].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 2].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[0, 2].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[0, 2].tick_params(axis='y', labelleft=False, left=False)
if n_turbine ==1:
    if type == 'no_inflow':
        axes[0, 2].set_title(' FNO ', fontsize=22)
    elif type == 'inflow':
        axes[0, 2].set_title(' CNN with inflow ', fontsize=22)
elif n_turbine ==3:
    if type == 'no_inflow':
        axes[0, 2].set_title(' Plain CNN ', fontsize=22)
    elif type == 'inflow':
        axes[0, 2].set_title(' CNN with inflow ', fontsize=22)

# ----------------------  绘制第二行  --------------------------#
# 绘制第一个子图
im2_1 = axes[1, 0].imshow(grid_u_cs_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[1, 0].set_xlim(200, 700)
axes[1, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

if n_turbine == 1:
    axes[1, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[1, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[1, 0].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[1, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)

# 绘制第二个子图
im2_2 = axes[1, 1].imshow(grid_u_true_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[1, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[1, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[1, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[1, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[1, 1].tick_params(axis='y', labelleft=False, left=False)

# 绘制第三个子图
im2_3 = axes[1, 2].imshow(grid_u_black_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[1, 2].set_xlim(200, 700)
if n_turbine == 1:
    axes[1, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[1, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 2].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 2].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[1, 2].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[1, 2].tick_params(axis='y', labelleft=False, left=False)

# ----------------------  绘制第三行  --------------------------#
# 绘制第一个子图
im3_1 = axes[2, 0].imshow(grid_u_cs_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[2, 0].set_xlim(200, 700)
axes[2, 0].tick_params(axis='both', labelsize=20)  # 字号设为12
# # ----------------------------------------------#
if n_turbine == 1:
    axes[2, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[2, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)

axes[2, 0].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[2, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


# 绘制第二个子图
im3_2 = axes[2, 1].imshow(grid_u_true_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[2, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[2, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[2, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)

axes[2, 1].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[2, 1].tick_params(axis='y', labelleft=False, left=False)

# 绘制第三个子图
im3_3 = axes[2, 2].imshow(grid_u_black_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[2, 2].set_xlim(200, 700)
if n_turbine == 1:
    axes[2, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[2, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 2].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 2].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[2, 2].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[2, 2].tick_params(axis='y', labelleft=False, left=False)

# ----------------------  绘制第四行  --------------------------#
# 绘制第一个子图
im3_1 = axes[3, 0].imshow(grid_u_cs_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[3, 0].set_xlim(200, 700)
axes[3, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

if n_turbine == 1:
    axes[3, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[3, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[3, 0].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[3, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


# 绘制第二个子图
im3_2 = axes[3, 1].imshow(grid_u_true_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower',
                          cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[3, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[3, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[3, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[3, 1].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[3, 1].tick_params(axis='y', labelleft=False, left=False)

# 绘制第三个子图
im3_3 = axes[3, 2].imshow(grid_u_black_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower',
                          cmap='coolwarm',
                          aspect='auto', vmin=vmin, vmax=vmax)
if type == 'inflow':
    axes[3, 2].set_xlim(200, 700)
if n_turbine == 1:
    axes[3, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[3, 2].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 2].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 2].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[3, 2].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[3, 2].tick_params(axis='y', labelleft=False, left=False)


# ------------------------- 调整布局 ---------------------#
plt.tight_layout(rect=[0.06, 0.12, 1, 0.98])
# ------------------------- 共享颜色条 --------------------#
# 获取子图位置信息
pos1 = axes[0, 0].get_position().bounds  # (左, 下, 宽, 高)
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

# ------------------------- 遍历：给每一行添加标注 ------------------------------#
for i, label in enumerate(row_labels):
    # 给第i行 添加左侧标题
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
# if n_turbine == 1:
#     if type == 'no_inflow':
#         plt.savefig(r"picture/compare_model/one_turbine/one_compare_cnn.pdf")
#     elif type == 'inflow':
#         plt.savefig(r"picture/compare_model/one_turbine/inflow/one_compare_cnn_inflow.pdf")
# elif n_turbine == 3:
#     if type == 'no_inflow':
#         plt.savefig(r"picture/compare_model/three_turbine/three_compare_cnn.pdf")
#     elif type == 'inflow':
#         plt.savefig(r"picture/compare_model/three_turbine/inflow/three_compare_cnn_inflow.pdf")
plt.show()

####################################################################################################################
#-----------------------------------------   误差图   --------------------------------------------------------------#
# 找出两个插值数据的最小值和最大值
vmin_error = min(np.nanmin(grid_u_cs_error_1), np.nanmin(grid_u_black_error_1), np.nanmin(grid_u_cs_error_2), np.nanmin(grid_u_black_error_2),
           np.nanmin(grid_u_cs_error_3), np.nanmin(grid_u_black_error_3), np.nanmin(grid_u_cs_error_end), np.nanmin(grid_u_black_error_end))
vmax_error = max(np.nanmax(grid_u_cs_error_1), np.nanmax(grid_u_black_error_1), np.nanmax(grid_u_cs_error_2), np.nanmax(grid_u_black_error_2),
           np.nanmax(grid_u_cs_error_3), np.nanmax(grid_u_black_error_3), np.nanmax(grid_u_cs_error_end), np.nanmax(grid_u_black_error_end))
vmin_error_1 = min(np.nanmin(grid_u_cs_error_1-grid_u_black_error_1), np.nanmin(grid_u_cs_error_2-grid_u_black_error_2),
                 np.nanmin(grid_u_cs_error_3-grid_u_black_error_3),  np.nanmin(grid_u_cs_error_end-grid_u_black_error_end))
vmax_error_1 = max(np.nanmax(grid_u_cs_error_1-grid_u_black_error_1), np.nanmax(grid_u_cs_error_2-np.nanmax(grid_u_black_error_2)),
                 np.nanmax(grid_u_cs_error_3-grid_u_black_error_3),np.nanmax(grid_u_cs_error_end-np.nanmax(grid_u_black_error_end)))

vmin_error=0.1
vmax_error = 1.2
vmin_error_1= -1
vmac_error_1 = 1
# ############# 画图(真实值和预测值同时显示)
# 计算坐标范围，确定纵横比
aspect_ratio = (max(x) - min(x)) / (max(y) - min(y))
fig, axes = plt.subplots(4, 2, figsize=(6 * aspect_ratio, 8))
# 调整子图水平间距和底部边距
plt.subplots_adjust(wspace=0.1, bottom=0.25)  # wspace控制子图间距，bottom预留颜色条空间
# 定义每行要显示的行标题
row_labels = [f"T+{pre_step_1}", f"T+{pre_step_2}", f"T+{pre_step_3}", f"T+{pre_step_end}"]

# ----------------------  绘制第一行  --------------------------#
# 绘制第一个子图
im1 = axes[0, 0].imshow(grid_u_cs_error_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                        aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[0, 0].set_xlim(200, 700)
axes[0, 0].tick_params(axis='both', labelsize=20)  # 字号设为12
if n_turbine == 1:
    axes[0, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[0, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[0, 0].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[0, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)
axes[0, 0].set_title('PPNN-cs', fontsize=22)

# 绘制第二个子图
im2 = axes[0, 1].imshow(grid_u_black_error_1.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                        aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[0, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[0, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[0, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[0, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[0, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[0, 1].tick_params(axis='y', labelleft=False, left=False)
if n_turbine == 1:
    if type == 'no_inflow':
        axes[0, 1].set_title('FNO ', fontsize=22)
    elif type == 'inflow':
        axes[0, 1].set_title('CNN with inflow ', fontsize=22)
elif n_turbine == 3:
    if type == 'no_inflow':
        axes[0, 1].set_title('Plain CNN ', fontsize=22)
    elif type == 'inflow':
        axes[0, 1].set_title('CNN with inflow ', fontsize=22)

# ----------------------  绘制第二行  --------------------------#
# 绘制第一个子图
im2_1 = axes[1, 0].imshow(grid_u_cs_error_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[1, 0].set_xlim(200, 700)
axes[1, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

if n_turbine == 1:
    axes[1, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[1, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[1, 0].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[1, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)

# 绘制第二个子图
im2_2 = axes[1, 1].imshow(grid_u_black_error_2.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[1, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[1, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[1, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[1, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[1, 1].tick_params(axis='x', labelbottom=False, bottom=False)  # 字号设为12
axes[1, 1].tick_params(axis='y', labelleft=False, left=False)


# ----------------------  绘制第三行  --------------------------#
# 绘制第一个子图
im3_1 = axes[2, 0].imshow(grid_u_cs_error_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[2, 0].set_xlim(200, 700)
axes[2, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

if n_turbine == 1:
    axes[2, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[2, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)

axes[2, 0].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[2, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


# 绘制第二个子图
im3_2 = axes[2, 1].imshow(grid_u_black_error_3.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[2, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[2, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[2, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[2, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)

axes[2, 1].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[2, 1].tick_params(axis='y', labelleft=False, left=False)



# ----------------------  绘制第四行  --------------------------#
# 绘制第一个子图
im3_1 = axes[3, 0].imshow(grid_u_cs_error_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower',
                          cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[3, 0].set_xlim(200, 700)
axes[3, 0].tick_params(axis='both', labelsize=20)  # 字号设为12

if n_turbine == 1:
    axes[3, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[3, 0].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 0].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 0].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[3, 0].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[3, 0].tick_params(axis='y', labelsize=20, labelleft=False, left=False)


# 绘制第二个子图
im3_2 = axes[3, 1].imshow(grid_u_black_error_end.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='coolwarm',
                          aspect='auto', vmin=vmin_error, vmax=vmax_error)
if type == 'inflow':
    axes[3, 1].set_xlim(200, 700)
if n_turbine == 1:
    axes[3, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
elif n_turbine == 3:
    axes[3, 1].plot([500, 500], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 1].plot([1100, 1100], [600 - 50, 600 + 50], color='black', linewidth=2)
    axes[3, 1].plot([1700, 1700], [600 - 50, 600 + 50], color='black', linewidth=2)
axes[3, 1].tick_params(axis='x', labelsize=20, labelbottom=False, bottom=False)  # 字号设为12
axes[3, 1].tick_params(axis='y', labelleft=False, left=False)

# ------------------------- 调整布局 ---------------------#
plt.tight_layout(rect=[0.06, 0.12, 1, 0.98])
# 获取子图位置信息
pos1 = axes[0, 0].get_position().bounds  # (左, 下, 宽, 高)
pos2 = axes[0, 1].get_position().bounds
# ----- 前两个子图的共享颜色条 ----
# 计算颜色条的左边界和宽度，使其横跨两个子图
cax_left = pos1[0] + 0.005
cax_width = pos2[0] + pos2[2] - pos1[0]- 0.01

# 设置颜色条的位置参数（距离底部0.1，高度0.03）
cax = fig.add_axes([cax_left, 0.1, cax_width, 0.03])

# 创建水平颜色条
cbar = fig.colorbar(im1, cax=cax, orientation='horizontal', label='error(m/s)')
cbar.set_label('Relative Error', fontsize=22)  # 设置颜色条标签
cbar.ax.tick_params(labelsize=20)  # 刻度标签字号14


# ------------------------- 遍历：给每一行添加标注 ------------------------------#
for i, label in enumerate(row_labels):
    # 给第i行 添加左侧标题
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
# if n_turbine == 1:
#     if type == 'no_inflow':
#         plt.savefig(r"picture/compare_model/one_turbine/one_compare_cnn_error.pdf")
#     elif type == 'inflow':
#         plt.savefig(r"picture/compare_model/one_turbine/inflow/one_compare_cnn_inflow_error.pdf")
# elif n_turbine == 3:
#     if type == 'no_inflow':
#         plt.savefig(r"picture/compare_model/three_turbine/three_compare_cnn_error.pdf")
#     elif type == 'inflow':
#         plt.savefig(r"picture/compare_model/three_turbine/inflow/three_compare_cnn_inflow_error.pdf")

plt.show()