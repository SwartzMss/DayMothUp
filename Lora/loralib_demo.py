# -*- coding: utf-8 -*-
import loralib as lora
import torch
import torch.nn as nn
from bigmodelvis import Visualization

in_features = 128
n_class = 2
d_dim = 64

# define a model that contains two lora linear layers
class Model(nn.Module):
    def __init__(self, in_feature, d_dim, n_class):
        super(Model, self).__init__()
        self.layer1 = lora.Linear(in_feature, d_dim, r=16)
        print(f"self.layer1.lora_A.shape = {self.layer1.lora_A.shape}")
        print(f"self.layer1.lora_B.shape = {self.layer1.lora_B.shape}")
        self.layer2 = lora.Linear(d_dim, n_class, r=16)
        #print(f"self.layer2.shape = {self.layer2.shape}")
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(f"x.shape() = {x.shape}")
        x = self.layer1(x)
        #print(f"self.layer1(x).shape() = {x.shape}")
        x = self.relu(x)
        #print(f"self.relu(x).shape() = {x.shape}")
        x = self.layer2(x)
        #print(f"self.layer2(x).shape() = {x.shape}")
        return self.log_softmax(x)
    '''
    x.shape() = torch.Size([16, 128])
    self.layer1(x).shape() = torch.Size([16, 64])
    self.relu(x).shape() = torch.Size([16, 64])
    self.layer2(x).shape() = torch.Size([16, 2])
    '''

    '''
    self.layer1.lora_A.shape = torch.Size([16, 128])
    self.layer1.lora_B.shape = torch.Size([64, 16])
    '''

    '''
    layer.scaling = 1/lora_rank 只是一个数值
    '''

    '''
    矩阵运算如下layer1：
    x.shape = [16, 128]
    weight = [64, 128]
    c = lora_B@lora_A = [64, 16] * [16, 128] = [64, 128]
    xx = weight + c*layer.scaling
    x @ xx.T = [16, 128] * [64 128].T = [16, 128] * [128, 64] = [16,64]
    '''

    '''
    默认只有LORA的那一层才需要进行backforwd
    layer1.weight torch.Size([64, 128]) cpu not requires_grad
    layer1.bias torch.Size([64]) cpu not requires_grad
    layer1.lora_A torch.Size([16, 128]) cpu requires_grad
    layer1.lora_B torch.Size([64, 16]) cpu requires_grad
    layer2.weight torch.Size([2, 64]) cpu not requires_grad
    layer2.bias torch.Size([2]) cpu not requires_grad
    layer2.lora_A torch.Size([16, 64]) cpu requires_grad
    layer2.lora_B torch.Size([2, 16]) cpu requires_grad
    '''

if __name__ == "__main__":
    # create a model
    # Add a pair of low-rank adaptation matrices with rank r=16
    model = Model(in_features, d_dim, n_class)
    Visualization(model).structure_graph()
    for name, param in model.named_parameters():
        if "lora" in name:
            print(name, param.shape, param.device, "requires_grad")
            param.requires_grad = True
        else:
            print(name, param.shape, param.device, "not requires_grad")
            param.requires_grad = True


    # fake some input data
    Visualization(model).structure_graph()
    x = torch.randn(16, in_features)
    y = torch.randint(0, n_class, (16,))

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == y).float().mean()
        if i % 10 == 0:
            print(f"i: {i}, loss: {loss.item():.3f}, acc: {acc:.3f}")
