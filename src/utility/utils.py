from functools import partial
from typing import Tuple
from io import StringIO
import torch
from torch.nn.functional import interpolate as interp
import numpy as np


def model_count(model):
    return sum(p.numel() for p in model.parameters())  # 计算神经网络模型中所有参数的总数


class mesh_convertor(object):  #上采样/下采样
    '''
        Class for fine/coarse mesh conversion
    '''

    def __init__(self, fmesh_size: float, cmesh_size: float, dim=1, align_corners=True) -> None:
        super().__init__()
        # self.down_ratio = cmesh_size/fmesh_size
        # self.up_ratio = fmesh_size/cmesh_size
        self.align_corners = align_corners
        downmode = 'linear' if dim == 1 else 'bilinear'  # 利用线性或者双线性函数进行下采样·
        self.fmesh_size = fmesh_size
        self.down = partial(interp, mode=downmode, size=cmesh_size, align_corners=align_corners, )  # 定义一个新的下采样函数
        self.up = self.up1d if dim == 1 else self.upnd  # 选择上采样类型

    def up1d(self, u):
        return interp(u.unsqueeze(1), mode='bicubic', size=self.fmesh_size, align_corners=self.align_corners, )[:, :, 0]
    # u 张量的第 1 维（即第二个维度）插入一个新的维度 ; 切片，选择第三维的第 0 个元素 ； 利用双三次插值进行上采样

    def upnd(self, u):
        return interp(u, mode='bicubic', size=self.fmesh_size, align_corners=self.align_corners, ) # 利用双三次插值进行上采样



def numpy2string(x, format='(%.6g %.6g %.0g)'):  # 将一个 NumPy 数组 x 转换成一个 字符串格式
    fakefile = StringIO()
    np.savetxt(fakefile, x, fmt=format)
    return fakefile.getvalue()


if __name__ == '__main__':
    from math import pi

    fx = torch.linspace(0, 2 * pi, 101)
    cx = torch.linspace(0, 2 * pi, 11)
    uf = torch.sin(fx).reshape(1, 1, -1)
    uc = torch.sin(cx).reshape(1, 1, -1)
    mcter = mesh_convertor(fmesh_size=fx.shape[-1], cmesh_size=cx.shape[-1])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(fx, uf.squeeze(), label='original')
    ax[0].plot(cx, mcter.down(uf).squeeze(), label='down')
    ax[1].plot(cx, uc.squeeze(), '--', label='original')
    ax[1].plot(fx, mcter.up(uc).squeeze(), label='up')
    ax[0].legend()
    ax[1].legend()
    plt.show()

