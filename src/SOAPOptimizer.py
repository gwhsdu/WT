import torch
import torch.optim as optim

class SOAPOptimizer:
    def __init__(self, model, lr=0.001,beta1=0.9, beta2=0.95 ):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2


        # 使用 PyTorch 自带的 Adam 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1,self.beta2))

        # 初始化 L 和 R（用于 SOAP 更新）
        self.L = []
        self.R = []
        self.QL = []  # 存储每层的 QL
        self.QR = []  # 存储每层的 QR

        # 对每个权重矩阵初始化 L 和 R
        for param in self.model.parameters():
            if param.ndimension() == 2:  # 如果是权重矩阵 (m, n)
                m, n = param.shape
                self.L.append(torch.zeros(m, m, device=param.device))  # 初始化 L 为 m × m 的零矩阵
                self.R.append(torch.zeros(n, n, device=param.device))  # 初始化 R 为 n × n 的零矩阵
                self.QL.append(None)  # 初始化 QL
                self.QR.append(None)  # 初始化 QR
            else:
                # 对于偏置，L 和 R 不进行初始化
                self.L.append(None)
                self.R.append(None)
                self.QL.append(None)
                self.QR.append(None)

    def step(self, grads):
        # Step 1: Project the gradient to the eigenspace (L, R)
        for i, (param, grad) in enumerate(zip(self.model.parameters(), grads)):
            # 只对权重执行 SOAP 更新
            if param.ndimension() == 2:  # 确保这是权重矩阵
                # print(f"grad[{i}] size: {grad.size()}")  # 输出 grad 的大小

                # 使用 SOAP 更新 L 和 R
                self.L[i] = self.beta2 * self.L[i] + (1 - self.beta2) * grad @ grad.T  # 更新L
                self.R[i] = self.beta2 * self.R[i] + (1 - self.beta2) * grad.T @ grad  # 更新R

                # 输出 L[i] 和 R[i] 的大小（维度）
                # print(f"Updated L[{i}] size: {self.L[i].size()}")  # 输出 L[i] 的大小
                # print(f"Updated R[{i}] size: {self.R[i].size()}")  # 输出 R[i] 的大小

                # 使用 QR 分解来得到 L 和 R 的特征分解
                self.QL[i], LambdaL = torch.linalg.qr(self.L[i])  # QL 是特征向量，LambdaL 是特征值
                self.QR[i], LambdaR = torch.linalg.qr(self.R[i])  # QR 是特征向量，LambdaR 是特征值

                # 输出 QL 和 QR 的大小（维度）
                # print(f"QL[{i}] size: {self.QL[i].size()}")  # 输出 QL 的大小
                # print(f"QR[{i}] size: {self.QR[i].size()}")  # 输出 QR 的大小

                # Step 2: Apply Adam update in the rotated space
                G_t = self.QL[i].T @ grad @ self.QR[i]  # 投影梯度到特征空间
                # G_t = G_t / (G_t.norm() + 1e-8)  # 归一化梯度

                # Step 3: 将变换后的梯度存回权重参数
                param.grad = G_t
            else:  # 偏置 (通常是 1D 向量)
                # 偏置的梯度直接使用，不做特征空间变换
                param.grad = grad

        # Step 3: 使用 PyTorch 自带的 Adam 更新
        self.optimizer.step()  # 更新偏置（Adam）

        # Step 3: Transform back to the original parameter space
        for i, param in enumerate(self.model.parameters()):
            if param.ndimension() == 2:  # 对于权重矩阵
                # print(f"grad[{i}] size: {param.data.size()}")  # 输出 grad 的大小
                # 将更新后的权重从特征空间转换回原始空间
                # QL, LambdaL = torch.linalg.qr(self.L[i])
                # QR, LambdaR = torch.linalg.qr(self.R[i])

                param.data = self.QL[i] @ param.data @ self.QR[i].T  # 将更新后的权重转回原始空间

    def zero_grad(self):
        # 重置梯度
        for param in self.model.parameters():
            param.grad = None