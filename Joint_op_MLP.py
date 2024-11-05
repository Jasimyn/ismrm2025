# oding = utf-8
# -*- coding:utf-8 -*-
# oding = utf-8
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import random
import torch.optim as optim
from op_func import *
from scipy.io import savemat
import math
import torch.autograd.forward_ad as fwad
import torch

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# -number of acquisitions we want to use
num_acqs = 5

# -representative tissue parameters to compute CRB for optimization
parameters = torch.tensor([[70e-3,700e-3,1],
                           [80e-3,1300e-3,1]])
#                             #t2s t1 m0

nparam,N = parameters.shape
tf = 128
#-initializing flip angle train with standard 4 degree flip angles
alpha = torch.ones((tf*num_acqs)) * 4 / 180 * math.pi
alpha.requires_grad = True

#setting losses and tracking for optimization

alpha_init = alpha.clone()

pW = torch.tensor([1,1,1]).to(device)
#-relative weighting of each representative tissue parameter

#-defining weighting matrix for each of the parameters we want to estimate
W = torch.zeros((N, N, nparam)).to(device)
for pp in range(nparam):
    for nn in range(N):
        W[nn,nn,pp] = 1 / parameters[pp,nn]**2


# 模拟输入数据和网络参数
input_fa = torch.ones((tf*num_acqs)) * 4 / 180 * math.pi  # used to optimize FAs
input_gap = torch.ones(num_acqs) * 0.9  # used to optimize gap
input_prep = torch.tensor([109.7e-3])  # used to optimize prep
input_signal = torch.cat((input_fa, input_gap, input_prep))
input_signal = input_signal.unsqueeze(0).to(device)
L = input_fa.size(0)
# 初始化网络和优化器
model = MLPWithAttention(seq_length=L, input_size=L+num_acqs+1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 0.001
loss_init = crb_loss_joint(parameters, W, pW, input_fa, input_gap, input_prep, N, num_acqs, nparam)
print('init_loss', loss_init)
# 训练循环
iterations = 500  # 假设迭代100次
results = {'input_signal': input_signal.cpu().numpy()}  # 输入信号只保存一次
losses = []
for epoch in range(iterations):
    optimizer.zero_grad()
    # 前向传播
    fa, gap, prep = model(input_signal)

    # 计算CRB损失
    loss_1 = crb_loss_joint(parameters, W, pW, fa, gap, prep, N, num_acqs, nparam)
    loss_2 = torch.sum(gap) * 15
    print(loss_1)
    print(loss_2)
    loss = loss_1 + loss_2

    # 反向传播并优化
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{iterations}, Loss: {loss.item()}')
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        # 将输出从GPU移回CPU并转换为numpy数组
        fa_np = fa.detach().cpu().numpy()
        gap_np = gap.detach().cpu().numpy()
        prep_np = prep.detach().cpu().numpy()
        output_np = np.concatenate((fa_np, gap_np, prep_np))
        # 在字典中保存输出，键为 'output_epoch_20', 'output_epoch_40', 等
        results[f'output_epoch_{epoch + 1}'] = output_np

savemat('joint_mlp.mat', results)
print(f'Saved all outputs to all_outputs.mat')


fa, gap, prep = model(input_signal)

# 将输出结果从GPU移到CPU并转换为numpy数组以便绘图
fa_np = fa.detach().cpu().numpy()
gap_np = gap.detach().cpu().numpy()
prep_np = prep.detach().cpu().numpy()
input_fa = input_fa.cpu().numpy().flatten()

# 绘制输入信号和输出信号对比
plt.figure(figsize=(12, 6))
# 绘制输入信号
plt.subplot(1, 2, 1)
plt.plot(input_fa)
plt.title("Input fa")
plt.xlabel("Index")
plt.ylabel("Value")

# 绘制MLP输出信号
plt.subplot(1, 2, 2)
plt.plot(fa_np)
plt.title("MLP Output fa")
plt.xlabel("Index")
plt.ylabel("Value")

# 显示图像
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
