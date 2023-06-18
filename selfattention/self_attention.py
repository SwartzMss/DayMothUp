import torch.nn as nn
import torch
import matplotlib.pyplot as plt


#dim=2, dk=2, dv=3
class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(Self_Attention, self).__init__()
        #scale = 1 除以 dk 的平方根
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk) #[2,2]
        self.k = nn.Linear(dim, dk) #[2,2]
        self.v = nn.Linear(dim, dv) #[2,3]


    def forward(self, x):
        q = self.q(x) #[1,4,2]
        k = self.k(x) #[1,4,2]
        v = self.v(x) #[1,4,3]
        #k.transpose(-2, -1): 这个操作是将键（K）的最后两个维度进行转置
        '''
        例如，如果你有一个形状为 (a, b, c, d) 的四维张量，并且你调用了 transpose(-2, -1)，
        那么这会返回一个形状为 (a, b, d, c) 的张量。这是因为 -2 和 -1 分别表示倒数第二个和最后一个维度，即 c 和 d。

        在你给出的代码中，k.transpose(-2, -1) 就是将键矩阵 k 的最后两个维度进行了交换。
        这是因为在计算查询（Q）和键（K）的点积时，我们希望能够对应到每个查询的每个键，而不是每个键的每个查询。通过这样做，我们可以确保得到正确的注意力分数矩阵。
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale

        '''
        这边的话 其实是q和其他的k进行运算了已经，因为数据是作为一个batch进来的。
        '''
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x


att = Self_Attention(dim=2, dk=2, dv=3)
# 1 是batch size, 4 代表4个数据，2代表每个数据是二维
x = torch.rand((1, 4, 2))
print(f"x = {x}")
output = att(x)
print(f"output.shape ={output.shape} ,value = {output}")
'''
output.shape =torch.Size([1, 4, 3]) ,value = tensor([[[ 0.5370, -0.7457,  0.2682],
         [ 0.5370, -0.7461,  0.2690],
         [ 0.5382, -0.7431,  0.2622],
         [ 0.5346, -0.7491,  0.2763]]], grad_fn=<UnsafeViewBackward0>)
'''
















# class MultiHead_Attention(nn.Module):
#     def __init__(self, dim, num_heads):
#
#         super(MultiHead_Attention, self).__init__()
#         self.num_heads = num_heads   # 2
#         head_dim = dim // num_heads   # 2
#         self.scale = head_dim ** -0.5   # 1
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# att = MultiHead_Attention(dim=768, num_heads=12)
# x = torch.rand((1, 197, 768))
# output = att(x)


