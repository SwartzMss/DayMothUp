import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


# 获取脚本文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取脚本文件的目录
script_dir = os.path.dirname(script_path)

# 设置当前工作目录为脚本文件的目录
os.chdir(script_dir)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层，该层接受3个通道的输入（对应彩色图像的RGB三个通道），输出6个通道，卷积核的大小为5x5。
        self.conv1 = nn.Conv2d(3, 6, 5) 
        # 定义一个最大池化层，池化窗口的大小为2x2。
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层，该层接受6个通道的输入（对应上一层的输出通道数），输出16个通道，卷积核的大小为5x5。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个全连接层，该层接受16 * 5 * 5个输入神经元（对应上一层的输出），输出120个神经元。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层，该层接受120个输入神经元（对应上一层的输出），输出84个神经元。
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层，该层接受84个输入神经元（对应上一层的输出），输出10个神经元（对应CIFAR-10数据集的10个类别）。
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 首先通过第一个卷积层对输入x进行卷积操作，然后对结果进行ReLU激活，最后进行最大池化。
        x = self.pool(F.relu(self.conv1(x)))
        # 然后通过第二个卷积层对输入x进行卷积操作，然后对结果进行ReLU激活，最后进行最大池化。
        x = self.pool(F.relu(self.conv2(x)))
        # 将上一步的结果x变形为一个二维的Tensor，第二维的大小为16 * 5 * 5，这是为了准备输入到全连接层。-1表示第一维的大小会自动计算，通常是批次的大小。
        '''
        x = x.view(-1, 16 * 5 * 5)这行代码的作用是对张量x进行重塑（reshaping），将它转换为一个新的形状。

        在这个特定的场景中，x是一个4D的张量，它的形状是[batch_size, 16, 5, 5]。这是由于前面的卷积层和池化层的操作，
        其中batch_size是每个批次的样本数量，16是通道数量，最后两个5是图像的高和宽。

        view函数的作用就是重塑张量的形状。在这个例子中，-1是一个特殊的值，表示该维度的大小将根据张量的总元素数量和其他维度的大小自动计算。
        在这个例子中，-1实际上会被替换为batch_size，因此x.view(-1, 16 * 5 * 5)的结果是一个形状为[batch_size, 16*5*5]的2D张量。
        这一步是必要的，因为全连接层（如self.fc1）期望的输入是一个2D张量，其中第一维是批次维度，第二维是特征维度。
        通过view操作，我们将3D的图像数据（16个通道，每个通道是一个5x5的图像）压平为1D的特征向量，以输入到全连接层进行分类。
        '''
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(net, trainloader, criterion, optimizer, epochs):
    for epoch in range(2):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        end_time = time.time()
        print('Epoch %d finished in %.3f seconds' % (epoch + 1, end_time - start_time))
    print('Finished Training')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on", device)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, trainloader, criterion, optimizer, epochs=2)