
import random
from types import SimpleNamespace
from gc import collect
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
import math
import time

from torch.utils.data import random_split
from torch.utils.data import Subset, DataLoader

import models
import rhs
from SOAPOptimizer import SOAPOptimizer  # 导入SOAP优化器
from EarlyStopper import EarlyStopper
from operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from utility.utils import mesh_convertor, model_count


if __name__ == '__main__':
    inputfile = sys.argv[1]
    params = SimpleNamespace(
        **yaml.load(open(inputfile), Loader=yaml.FullLoader))
    random.seed(params.seed)  #####！！！！！！！！修改 ！！！！！！！！！！！！！
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)


    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    device = torch.device(params.device)
    feature_size = params.finemeshsize

    timesteps = params.timesteps


    dataonly = 0
    n_turbine = 3


    if n_turbine==1:
        # Ct_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        Ct_values = [0.0, 1.0, 2.0]
        Ct = torch.tensor(Ct_values, device=device)

        mu = Ct
        mu = mu.reshape(-1, 1, 1, 1)
        mus = mu.repeat(1, timesteps, 1, 1).reshape(-1, 1, 1, 1)

        mutest = mu[0:params.num_para]

    elif n_turbine ==3:
        aa = torch.load("data/three_turbine/aa_optimization_periodic.pt")
        mu = aa
        print("mu.shape:", mu.shape)
        mu = mu.reshape(-1, 1, 3, 1).float()
        mus = mu.repeat(1, timesteps, 1, 1).reshape(-1, 3, 1, 1).to(device)
        print("mus.device:",mus.device)
        print("mus.dtype:",mus.dtype)

        mutest = mu[0:params.num_para].reshape(-1, 3, 1, 1).to(device)
        print("mutest.shape:", mutest.shape)
        print("mutest:",mutest)

    #########################################################################################################


    testIDs = torch.linspace(0, params.num_para - 1, params.num_para, device=device, dtype=torch.long)



    def padbcx(uinner):
        u_left = 2 * uinner[:, :, 0:1] - uinner[:, :, 1:2]  # 左边界外推
        u_right = 2 * uinner[:, :, -1:] - uinner[:, :, -2:-1]  # 右边界外推
        u_padded = torch.cat((u_left, uinner, u_right), dim=2)
        return u_padded


    def padbcy(uinner):
        u_left = 2 * uinner[:, :,:, 0:1] - uinner[:, :,:, 1:2]  # 左边界外推
        u_right = 2 * uinner[:, :, :,-1:] - uinner[:, :,:, -2:-1]  # 右边界外推
        u_padded = torch.cat((u_left, uinner, u_right), dim=3)
        return u_padded

    def padBC_rd(u):
        tmp = torch.cat((u, u[:, :, :, :1]), dim=3)
        return torch.cat((tmp, tmp[:, :, :1]), dim=2)

    #############################################################################################


    if params.pde:
        rhsu = getattr(rhs, params.rhsu)
        rhsv = getattr(rhs, params.rhsv)
        rhsw = getattr(rhs, params.rhsw)

        d2udx2 = d2udx2_2D(accuracy=2, device=device)
        d2udy2 = d2udy2_2D(accuracy=2, device=device)
        dudx = dudx_2D(accuracy=1, device=device)
        dudy = dudy_2D(accuracy=1, device=device)
        cmesh = params.coarsemeshsize
        finemesh = params.finemeshsize
        dx = params.length[0] / (cmesh[0]-1)
        dy = params.length[1] / (cmesh[1]-1)
        mcvter = mesh_convertor(feature_size, cmesh, dim=2)
        dx2 = dx ** 2
        dy2 = dy ** 2
        dt = params.dt

        delta=(20*12*(950/96))**(1/3)

        print('\nCFL: {}\n'.format(dt / math.sqrt(dx*dy)))


        def pde_du(u, mu) -> torch.Tensor:
            u1 = mcvter.down(u[:, :1])[:, :, :, :]  # 下采样
            v1 = mcvter.down(u[:, 1:2])[:, :, :, :]  # 下采样
            w1 = mcvter.down(u[:, 2:3])[:, :, :, :]  # 下采样


            ux = padbcx(u1)
            uy = padbcy(u1)
            vx = padbcx(v1)
            vy = padbcy(v1)
            wx = padbcx(w1)
            wy = padbcy(w1)

            return torch.cat(
                (mcvter.up(
                        dt * rhsu(u1, v1, w1, mu, dudx, dudy, d2udx2, d2udy2, ux, uy, vx, vy, wx, wy, delta, dx, dy,
                                  dx2, dy2)
                ), \
                 mcvter.up(
                         dt * rhsv(u1, v1, w1, mu, dudx, dudy, d2udx2, d2udy2, ux, uy, vx, vy, wx, wy, delta, dx, dy,
                                   dx2, dy2)
                 ), \
                 mcvter.up(
                         dt * rhsw(u1, v1, w1, mu, dudx, dudy, d2udx2, d2udy2, ux, uy, vx, vy, wx, wy, delta, dx, dy,
                                   dx2, dy2)
                 )), dim=1)


            ############################################################################################################

    EPOCH = int(params.epochs) + 1
    BATCH_SIZE = int(params.batchsize)
    ########## cpu
    fdata: torch.Tensor = torch.load(params.datafile, map_location='cpu', ) \
                              [:, params.datatimestart:params.datatimestart + params.timesteps + 1].detach().to(
        torch.float)[:]
    fdata=fdata[:,:1500]
    ######### cpu
    testIDs = testIDs.detach().cpu()
    #####
    init = fdata[testIDs, 0]
    u_init_eva=fdata[testIDs, :-20]
    print("u_init_eva.shape:",u_init_eva.shape)
    ############## cpu
    label = fdata[testIDs, 1:, ].detach().cpu()
    data_u0 = fdata[:, :-1].reshape(-1, 3, feature_size[0], feature_size[1])
    data_du = (fdata[:, 1:] - fdata[:, :-1, ]).reshape(-1, 3, feature_size[0],
                                                       feature_size[1])

    fdata = []
    del fdata
    collect()


    def magnitude(x):
        return torch.sqrt(torch.sum(x ** 2, dim=-3))


    def add_plot(p, l=None):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        p0 = ax[0].pcolormesh(p, clim=(l.min(), l.max()))
        fig.colorbar(p0, ax=ax[0])
        if l is not None:
            p2 = ax[1].pcolormesh(l, clim=(l.min(), l.max()))
            fig.colorbar(p2, ax=ax[1])
        return fig


    class myset(torch.utils.data.Dataset):
        def __init__(self):
            global data_u0, data_du

            self.inmean = data_u0[:, :, :, :].mean(dim=(0, 2, 3), keepdim=True)
            self.instd = data_u0[:, :, :, :].std(dim=(0, 2, 3), keepdim=True)
            self.u0_normd = (data_u0[:, :, :, :] - self.inmean) / self.instd
            ####################################################################################
            if params.noiseinject:
                data_u0 += 0.03 * self.instd  * torch.randn_like(data_u0)

            if params.pde:
                pdeu = pde_du(data_u0.to(device), mus).cpu()

                self.pdemean = pdeu.mean(dim=(0, 2, 3), keepdim=True)
                self.pdestd = pdeu.std(dim=(0, 2, 3), keepdim=True)
                self.pdeu_normd = (pdeu[:, :, :, :] - self.pdemean) / self.pdestd

                du = (data_du - pdeu)[:, :, :, :]


                ###########################################################################

                del pdeu
                collect()
                del data_du
                collect()
            else:
                collect()
                du = data_du[:, :, :, :]

                del data_du
                collect()

            self.outmean = du.mean(dim=(0, 2, 3), keepdim=True)
            self.outstd = du.std(dim=(0, 2, 3), keepdim=True)
            self.du_normd = (du - self.outmean) / self.outstd
            ######### cpu
            self.mu = mus.cpu()

            self.mumean = self.mu.mean()
            self.mustd = self.mu.std()
            self.mu_normd = (self.mu - self.mu.mean()) / self.mu.std()

        def __getitem__(self, index):
            if params.pde:
                return self.u0_normd[index], self.du_normd[index], self.mu_normd[index], self.pdeu_normd[index]
            else:
                return self.u0_normd[index], self.du_normd[index], self.mu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]


    dataset = myset()
    inmean, instd = dataset.inmean.to(device), dataset.instd.to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    mumean, mustd = dataset.mumean.to(device), dataset.mustd.to(device)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(params.tensorboarddir)

    torch.save(inmean, os.path.join(params.tensorboarddir, 'inmean.pt'))
    torch.save(instd, os.path.join(params.tensorboarddir, 'instd.pt'))
    torch.save(outmean, os.path.join(params.tensorboarddir, 'outmean.pt'))
    torch.save(outstd, os.path.join(params.tensorboarddir, 'outstd.pt'))
    torch.save(mumean, os.path.join(params.tensorboarddir, 'parsmean.pt'))
    torch.save(mustd, os.path.join(params.tensorboarddir, 'parsstd.pt'))

    if params.pde:
        pdemean, pdestd = dataset.pdemean.to(device), dataset.pdestd.to(device)
        torch.save(pdemean, os.path.join(params.tensorboarddir, 'pdemean.pt'))
        torch.save(pdestd, os.path.join(params.tensorboarddir, 'pdestd.pt'))

        #############################  划分数据集 ################################
        # 1. 划分数据集
    data_generator = torch.Generator().manual_seed(params.seed)
    total_size = len(dataset)

    data_condition = params.num_para
    data_timepoint = total_size / data_condition
    train_indices = (torch.arange(data_condition).reshape(-1, 1) * data_timepoint + torch.arange(
        0.7 * data_timepoint).reshape(1, -1)).flatten().long()
    val_indices = (torch.arange(data_condition).reshape(-1, 1) * data_timepoint + torch.arange(0.7 * data_timepoint,
                                                                                               0.85 * data_timepoint).reshape(
        1, -1)).flatten().long()
    test_indices = (torch.arange(data_condition).reshape(-1, 1) * data_timepoint + torch.arange(0.85 * data_timepoint,
                                                                                                data_timepoint).reshape(
        1, -1)).flatten().long()



    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    test_label = label.reshape(-1, 3, feature_size[0], feature_size[1])[test_indices]
    test_u0 = data_u0[test_indices]
    test_mu = mus[test_indices]
    del data_u0  #
    collect()

    # 2. 为每个子集创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        generator=data_generator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    # 3. 验证划分是否正确
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"总大小: {len(train_dataset) + len(val_dataset) + len(test_dataset)} (原始: {total_size})")





    model = getattr(models, params.network)().to(device)
    print("模型前几个参数的初始值：")
    for i, param in enumerate(model.parameters()):
        if i < 2:
            print(f"参数{i + 1}的前5个元素：{param.data.flatten()[:5]}")
        else:
            break

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = SOAPOptimizer(model, lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min', factor=0.8, patience=50,
                                                           cooldown=100, min_lr=5e-5)
    early_stopper = EarlyStopper(patience=50, delta=0.0001, mode='min')
    criterier = nn.MSELoss()
    criterier1 = nn.MSELoss()
    test_error_best = 0.05
    if n_turbine==1:
        save_dir = "data/postdata/one_turbine"
    elif n_turbine==3:
            save_dir = "data/postdata/three_turbine"
    train_losses_stor = []
    val_losses_stor = []

    start_time = time.time()

    for i in range(EPOCH):
        loshis = 0
        counter = 0

        for data in train_loader:

            if params.pde:
                u0, du, mu, pdeu = data
                u0, du, mu, pdeu = u0.to(device), du.to(device), mu.to(device), pdeu.to(device)
            else:
                u0, du, mu = data
                u0, du, mu = u0.to(device), du.to(device), mu.to(device)

            if params.pde:
                u_p = model(u0, mu, pdeu)
            else:
                u_p = model(u0, mu)

            loss = criterier(u_p, du)

            optimizer.zero_grad()
            loss.backward()
            grads = [param.grad for param in model.parameters()]
            loshis += loss.item()
            optimizer.step(grads)
            counter += 1

        writer.add_scalar('loss', loshis / counter, i)
        scheduler.step(loshis / counter)

        if i % 10 == 0:
            print('loss: {0:4f}\t epoch:{1:d}'.format(loshis / counter, i))
            train_losses_stor.append(loshis / counter)

            model.eval()
            test_results=[]

            timesteps1 = 10
            pre_steps= u_init_eva.shape[1]
            for j in range(0,pre_steps,30):
                u=u_init_eva[:,j].to(device)
                test_re = []
                for _ in range(timesteps1):

                    if params.pde:
                        pdeuu = pde_du(u, mutest).to(device)
                        u_tmp = model((u[:, :, :, :] - inmean) / instd,
                                           (mutest - mumean) / mustd,
                                           (pdeuu[:, :, :, :] - pdemean) / pdestd) * outstd + outmean \
                            + u + pdeuu

                    else:

                        u_tmp = model((u[:, :, :, :] - inmean) / instd,
                                           (mutest - mumean) / mustd) * outstd + outmean \
                            + u
                    #########################################################################################

                    u = u_tmp
                    test_re.append(u.detach())

                test_results.append(test_re)

            model.train()

            
            test_error_total=0
            for j1 in range(len(test_results)):
                error_j=test_results[j1]
                ##### cpu
                error_j = torch.stack(error_j, dim=1).cpu()

                test_error = torch.norm(error_j[:, -1] - label[:, j1 * 30 + timesteps1 - 1], p=2,
                                        dim=(1, 2, 3)) / torch.norm(label[:, j1 * 30 + timesteps1 - 1], p=2,
                                                                    dim=(1, 2, 3))

                test_error_mean = torch.mean(test_error)

                test_error_total += test_error_mean

            test_error_m=test_error_total/len(test_results)
            print("test_error_mean:",test_error_m)

            
            

            writer.add_scalar('rel_error', test_error_m, i)

            if test_error_m < test_error_best:
                test_error_best = test_error_m
                torch.save(model.state_dict(), params.modelsavepath)

        #############################   验证阶段 #############################
        
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data in val_loader:

                if params.pde:
                    u0, du, mu, pdeu = data
                    u0, du, mu, pdeu = u0.to(device), du.to(device), mu.to(device), pdeu.to(device)
                else:
                    u0, du, mu = data
                    u0, du, mu = u0.to(device), du.to(device), mu.to(device)


                if params.pde:
                    u_p = model(u0, mu, pdeu)
                else:
                    u_p = model(u0, mu)
                loss = criterier(u_p, du)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        if i % 10 == 0:
            print('val_loss:',val_loss)
            val_losses_stor.append(val_loss)
        # 早停判断
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {i}")
            break


        ################################################################################
    writer.close()
    end_time = time.time()

    total_time = end_time - start_time

    print("Total training time: {:.2f} seconds".format(total_time))

################################   测试阶段   20250313  #####################################
    # 自定义配对函数
    def paired_loader(loader, *args):
        """
        loader: DataLoader，返回多个变量的元组
        *args: 外部序列 (test_label, test_u0, test_mu)
        返回: (data_tuple, labels, test_u01, test_mu1)
        """
        for i, data_tuple in enumerate(loader):  # 接收整个元组
            start_idx = i * loader.batch_size
            end_idx = min((i + 1) * loader.batch_size, len(loader.dataset))
            batch_data = [data_tuple]  # 保持元组结构
            for seq in args:
                batch_data.append(seq[start_idx:end_idx])
            yield tuple(batch_data)
    ##############################################################
    model.eval()
    correct = 0
    accurate = 0
    total = 0
    correct_u = 0
    accurate_u = 0
    accurate_v = 0
    accurate_w = 0
    u_pre_all = []
    labels_all = []
    test_mu_all = []
    with torch.no_grad():
        for data,lables,test_u01,test_mu1 in paired_loader(test_loader,test_label,test_u0,test_mu):

            if params.pde:
                u0, du, mu, pdeu = data
                u0, du, mu, pdeu = u0.to(device), du.to(device), mu.to(device), pdeu.to(device)
            else:
                u0, du, mu = data
                u0, du, mu = u0.to(device), du.to(device), mu.to(device)



            if params.pde:
                u_p = model(u0, mu, pdeu)
                test_pdeu = pde_du(test_u01.to(device), test_mu1)
                test_u01 = test_u01.to(device)
                u_pre = test_u01 + test_pdeu + u_p * outstd + outmean
            else:
                u_p = model(u0, mu)
                test_u01 = test_u01.to(device)
                u_pre = test_u01 + u_p * outstd + outmean


            lables = lables.to(device)
            correct = math.sqrt(criterier(u_pre, lables)/criterier(lables, torch.zeros_like(lables)))
            correct_u = math.sqrt(criterier(u_pre[:,0], lables[:,0])/criterier(lables[:,0], torch.zeros_like(lables[:,0])))
            correct_v = math.sqrt(criterier(u_pre[:,1], lables[:,1])/criterier(lables[:,1], torch.zeros_like(lables[:,1])))
            correct_w = math.sqrt(criterier(u_pre[:,2], lables[:,2])/criterier(lables[:,2], torch.zeros_like(lables[:,2])))
            accurate += correct
            accurate_u += correct_u
            accurate_v += correct_v
            accurate_w += correct_w
            u_pre_all.append(u_pre)
            labels_all.append(lables)
            test_mu_all.append(test_mu1)
    u_pre_tensor = torch.cat(u_pre_all, dim=0)
    labels_tensor = torch.cat(labels_all, dim=0)
    test_mu_tensor = torch.cat(test_mu_all, dim=0)
    accurate = accurate / len(test_loader)
    print("accurate:",accurate)
    print("accurate_u:",accurate_u / len(test_loader))
    print("accurate_v:",accurate_v / len(test_loader))
    print("accurate_w:",accurate_w / len(test_loader))

    if n_turbine==1:
        torch.save(u_pre_tensor, os.path.join(save_dir,"u_pre_one_ori.pt"))
        torch.save(labels_tensor, os.path.join(save_dir,"labels_one_ori.pt"))
        torch.save(test_mu_tensor, os.path.join(save_dir,"test_mu_one_ori.pt"))
        np.save(os.path.join(save_dir, 'train_one_losses.npy'), train_losses_stor)
        np.save(os.path.join(save_dir, 'val_one_losses.npy'), val_losses_stor)
    elif n_turbine==3:
        torch.save(u_pre_tensor, os.path.join(save_dir, "u_pre_three_ori.pt"))
        torch.save(labels_tensor, os.path.join(save_dir, "labels_three_ori.pt"))
        torch.save(test_mu_tensor, os.path.join(save_dir, "test_mu_three_ori.pt"))
        np.save(os.path.join(save_dir, 'train_three_losses.npy'), train_losses_stor)
        np.save(os.path.join(save_dir, 'val_three_losses.npy'), val_losses_stor)

