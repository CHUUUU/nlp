import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 30, 5)
        self.fc1 = nn.Linear(30*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc1(x)
        out = F.relu(x, inplace=True)

        return out


if __name__ == '__main__':
    net = network()
    optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(3):
        for i, data in enumerate(trainloader, 0):
            optim.zero_grad()
            input, label = data
            input, label = Variable(input), Variable(label)

            optim.zero_grad()  # model weight 의 grad 값 0으로 set
            out = net(input)
            loss = loss_func(out, label)
            loss.backward() #  model weight 에 loss에 대한 grad 만듦
            optim.step()  # model wight 에 grad 를 update


# optimizer.zero_grad():
# Pytorch는 gradient를 loss.backward()를 통해 이전 gradient에 누적하여 계산한다.
# 이전 grad 를 누적 시킬 거면 누적 시킬만큼만 사용하고 optimizer.zero_grad()를 사용한다.
# 누적될 필요 없는 모델에서는 model에 input를 통과시키기 전 optimizer.zero_grad()를 한번 호출해 주면 된다.
# 그러면 optimzer 에 준 모든 파라미터에 속성으로 저장되어 있던 grad 가 0이 된다.
