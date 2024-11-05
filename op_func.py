import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.autograd.forward_ad as fwad
import math
from simulation import *


# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self, input_size=620, hidden_size1=300, hidden_size2=100, output_size=620):
        super(MLP, self).__init__()
        # 定义 MLP 网络层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = torch.sigmoid(out)  # sigmoid 输出
        out = torch.sigmoid(out) * 4 / 180 * math.pi + 2 / 180 * math.pi  # sigmoid 输出
        return out


class MLP0(nn.Module):
    def __init__(self, input_size=620, hidden_size=100, output_size=620):
        super(MLP0, self).__init__()
        # 定义 MLP 网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out) * 4 / 180 * math.pi + 2 / 180 * math.pi  # sigmoid 输出
        # out = torch.sigmoid(out) * 200e-3 + 800e-3
        return out


class MLPWithAttention(nn.Module):
    def __init__(self, input_size=626, hidden_size1=300, hidden_size2=100, num_heads=4, seq_length=620):
        super(MLPWithAttention, self).__init__()

        # 定义 MLP 部分
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size2, num_heads=num_heads, batch_first=True)

        # 输出层
        self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4_fa = nn.Linear(hidden_size1, seq_length)  # 输出 620 长度的信号
        self.fc4_gap = nn.Linear(hidden_size1, 5)  # 输出 5 长度的参数
        self.fc4_prep = nn.Linear(hidden_size1, 1)  # 输出 1 长度的参数

    def forward(self, x):
        # MLP 前向传播部分
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)

        # 输入到多头注意力机制
        # 输入的维度需要适配 attention 的要求 (batch_size, seq_length, hidden_size2)
        # batch_size = x.size(0)
        # attn_input = out.unsqueeze(1)  # (batch_size, 1, hidden_size2) 扩展为序列长度为 1
        # attn_output, _ = self.multihead_attn(attn_in~put, attn_input, attn_input)
        # out = attn_output.squeeze(1)  # 去掉不必要的维度

        # MLP 后续层
        out = self.fc3(out)
        out = self.relu(out)

        # 输出三个部分
        signal_output = self.fc4_fa(out)  # 620 长度的信号
        params1_output = self.fc4_gap(out)  # 5 长度的参数
        params2_output = self.fc4_prep(out)  # 1 长度的参数

        # 输出值在 [0,1] 之间，可以通过 sigmoid 限制
        signal_output = torch.sigmoid(signal_output) * 4 / 180 * math.pi + 2 / 180 * math.pi
        params1_output = torch.sigmoid(params1_output) * 250e-3 + 750e-3
        params2_output = torch.sigmoid(params2_output) * 40e-3 + 80e-3

        return signal_output, params1_output, params2_output


def crb_loss(parameters, W, pW, para2op, N, num_acqs, nparam):

    total_crb = 0
    # 遍历每个参数集，计算CRB
    for pp in range(nparam):
        primal = parameters[pp, :].clone().requires_grad_()
        tangs = torch.eye(N)

        fwd_jac = []

        with fwad.dual_level():
            # 对每个输入进行前向传播，计算雅可比矩阵
            for tang in tangs:
                dual_input = fwad.make_dual(primal, tang)
                dual_output = simulate_alpha(para2op, dual_input, num_acqs)

                # 获取雅可比矩阵的列
                jacobian_column = fwad.unpack_dual(dual_output).tangent
                fwd_jac.append(jacobian_column)

        fwd_jac = torch.stack(fwd_jac).T
        fim = W[:, :, pp] @ torch.inverse(fwd_jac.T @ fwd_jac)

        # 计算CRB并加权
        crb = torch.real(torch.trace(fim)) * pW[pp]

        total_crb += crb

    # 返回总损失值（总CRB）
    return total_crb


def crb_loss_joint(parameters, W, pW, fa, gap, prep, N, num_acqs, nparam):

    total_crb = 0
    # 遍历每个参数集，计算CRB
    for pp in range(nparam):
        primal = parameters[pp, :].clone().requires_grad_()
        tangs = torch.eye(N)

        fwd_jac = []

        with fwad.dual_level():
            # 对每个输入进行前向传播，计算雅可比矩阵
            for tang in tangs:
                dual_input = fwad.make_dual(primal, tang)
                dual_output = simulate_joint(fa, gap, prep, dual_input, num_acqs)

                # 获取雅可比矩阵的列
                jacobian_column = fwad.unpack_dual(dual_output).tangent
                fwd_jac.append(jacobian_column)

        fwd_jac = torch.stack(fwd_jac).T
        fim = W[:, :, pp] @ torch.inverse(fwd_jac.T @ fwd_jac)

        # 计算CRB并加权
        crb = torch.real(torch.trace(fim)) * pW[pp]

        total_crb += crb

    # 返回总损失值（总CRB）
    return total_crb