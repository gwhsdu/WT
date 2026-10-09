import torch 
import torch.nn as nn


def gen_net(layers,insize,outsize,hsize):
    l = [nn.Linear(insize,hsize),]  #全连接层。这个层将输入特征从 insize 个映射到 hsize 个。
    for _ in range(layers):   #循环将运行 layers 次，每次迭代都会向列表 l 中添加一个新的层。
        l.append(lblock(hsize))
    l.append(nn.Linear(hsize, outsize))
    return nn.Sequential(*l)


class lblock(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)  #归一化
    
    def forward(self, x):
        return self.ln(self.net(x)) + x


class pmlp(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 hidden_layers = 2,
                 ) -> None:
        super().__init__()
        self.pnet = gen_net(0, 1, hidden_size, hidden_size)
        # self.pdenet = gen_net(h0, input_size, hidden_size, hidden_size)
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.convertnet = gen_net(2, 3, 1, 4)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, p):
        return self.hnet(
                    torch.stack(
                        (self.pnet(p), 
                         self.u0net(u0)
                        ),dim=-1
                    ).mean(dim=-1))  #前向传播


class mlpnop(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 p_h_layers = 0,
                 hidden_layers = 3,) -> None:
        super().__init__()
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, mu):
        return self.hnet(self.u0net(u0))

class cblock(nn.Module):
    def __init__(self,hc,ksize,feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc,hc,ksize,padding=(ksize)//2),
            nn.ReLU(),
            nn.Conv2d(hc,hc,ksize,padding=(ksize)//2),
        )
        self.ln = nn.LayerNorm([hc]+feature_size)
        
    def forward(self,x):
        return self.ln(self.net(x)) + x  # 残差连接


class cnet(nn.Module):
    def __init__(self, hc, ksize, feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc, hc, ksize, padding=(ksize) // 2),
            nn.ReLU(),
            nn.Conv2d(hc, hc, ksize, padding=(ksize) // 2),
        )
        self.ln = nn.LayerNorm([hc] + feature_size)

    def forward(self, x):
        return self.ln(self.net(x)) + x


class cnn2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(4, 48, 6, stride=2, padding=2),
            # nn.Conv2d(6, 48, 6, stride=2, padding=2),
         ###########
            nn.Conv2d(48, 96, 5, stride=1, padding=2),
            nn.ReLU(),


            ######## one_turbine ################
            cblock(96, 5, [40, 16]),  # [58,16]  原[40,16]
            cblock(96, 5, [40, 16]),
            cblock(96, 5, [40, 16]),  # 卷积＋残差
            #--------------------------------------------------------------#
            # cblock(96, 5, [58, 16]), #[58,16]  原[40,16]
            # cblock(96, 5, [58, 16]),
            # cblock(96, 5, [58, 16]),  # 卷积＋残差
            nn.PixelShuffle(2),  # 185 #将输入张量中的元素重新排列，以实现上采样的效果.4 是 2 的平方（2^2），这意味着每个维度（高度和宽度）将分别放大 2
            # #######################   非周期形成 53*39  ##################################
            nn.Conv2d(24, 3, 4, padding=[2, 2]),
            # nn.Conv2d(24, 3, [5,4], padding=[2, 2]),



        )

        ############################   非周期  20250313  ######################################
        self.cw = nn.Parameter(torch.randn(1,1,1,33)) #表示可学习的参数,元素是从标准正态分布中随机初始化的,可学习的参数的维度为 (1, 1, 1, 256)
        self.rw = nn.Parameter(torch.randn(1,1,81,1))
        # self.rw = nn.Parameter(torch.randn(1, 1, 116, 1))
        ########################################################################################

    def forward(self,u0,mu,pdeu=None):
        return self.net(torch.cat((u0,mu*self.rw@self.cw),dim=1)) #@ 表示矩阵乘法（或者称为“点积”）
                                                                        #self.net：这是网络的一个成员变量,会接收由 torch.cat 生成的拼接张量作为输入，并返回输出
                                                                        # pdeu=None：这是 forward 方法的一个可选参数。它在这个方法中没有被使用



###########################################################################################################################################
#####################################   LESGO的CNN2d    ################################################
class cnn2d_pde(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            # nn.Conv2d(7, 48, 6, stride=2, padding=2),
            nn.Conv2d(9, 48, 6, stride=2, padding=2),
         ###########
            nn.Conv2d(48, 96, 5, stride=1, padding=2),
            nn.ReLU(),

            ######## one_turbine ################
            # cblock(96, 5, [40, 16]),  # [58,16]  原[40,16]
            # cblock(96, 5, [40, 16]),
            # cblock(96, 5, [40, 16]),  # 卷积＋残差
            # --------------------------------------------------------------#
            cblock(96, 5, [58, 16]), #[58,16]  原[40,16]
            cblock(96, 5, [58, 16]),
            cblock(96, 5, [58, 16]),  # 卷积＋残差
            nn.PixelShuffle(2),  # 185 #将输入张量中的元素重新排列，以实现上采样的效果.4 是 2 的平方（2^2），这意味着每个维度（高度和宽度）将分别放大 2
            #######################   非周期形成 53*39  ##################################
            # nn.Conv2d(24, 3, 4, padding=[2, 2]),
            nn.Conv2d(24, 3, [5,4], padding=[2, 2]),



        )

        ############################   非周期  20250313  ######################################
        self.cw = nn.Parameter(torch.randn(1,1,1,33)) #表示可学习的参数,元素是从标准正态分布中随机初始化的,可学习的参数的维度为 (1, 1, 1, 256)
        # self.rw = nn.Parameter(torch.randn(1,1,81,1))
        self.rw = nn.Parameter(torch.randn(1, 1, 116, 1))
        ########################################################################################

    def forward(self,u0,mu,pdeu=None):
        U = torch.cat((u0,pdeu),dim=1)
        return self.net(torch.cat((U,mu*self.rw@self.cw),dim=1)) #@ 表示矩阵乘法（或者称为“点积”）
                                                                        #self.net：这是网络的一个成员变量,会接收由 torch.cat 生成的拼接张量作为输入，并返回输出
                                                                        # pdeu=None：这是 forward 方法的一个可选参数。它在这个方法中没有被使用




    
        

