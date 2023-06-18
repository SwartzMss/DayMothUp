from math import sqrt
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim_in, d_model, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()

        self.dim_in = dim_in # 2 每个输入的维度
        self.d_model = d_model # 6 如果我们使用self attention的情况下 qkv的总的向量的长度 
        self.num_heads = num_heads # 3 三个head(单个qkv),所以每个head的维度就是6/3=2

        # 维度必须能被num_head 整除
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"

        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, d_model)
        self.linear_k = nn.Linear(dim_in, d_model)
        self.linear_v = nn.Linear(dim_in, d_model)
        self.scale = 1 / sqrt(d_model // num_heads)

        # 最后的线性层
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.d_model // nh  # dim_k of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)

        dist = torch.matmul(q, k.transpose(2, 3)) * self.scale  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.d_model)  # batch, n, dim_v

        # 最后通过一个线性层进行变换
        output = self.fc(att)

        return output


# 1 是batch size, 4 代表4个数据，2代表每个数据是二维
x = torch.rand((1, 4, 2))
print(f"x = {x}")
print(f"x.shape[2] = {x.shape[2]}")
multi_head_att = MultiHeadSelfAttention(x.shape[2], 6, 3)  # (6, 3)
output = multi_head_att(x)
print(f"output.shape ={output.shape} ,value = {output}")
'''
output.shape =torch.Size([1, 4, 6]) ,value = tensor([[[0.7109, 0.2039, 0.2775, 0.9348, 0.0442, 0.0524],
         [0.7154, 0.2009, 0.2748, 0.9421, 0.0451, 0.0505],
         [0.7109, 0.2040, 0.2775, 0.9346, 0.0441, 0.0525],
         [0.7113, 0.2052, 0.2789, 0.9353, 0.0419, 0.0540]]],
       grad_fn=<ViewBackward0>)
'''