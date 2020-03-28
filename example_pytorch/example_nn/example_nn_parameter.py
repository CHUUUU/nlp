import torch
import torch.nn as nn

class model_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(300, 10))
        self.bias = nn.Parameter(torch.zeros(10))
    def forward(self, x):
        return x @ self.weights + self.bias


class model_two(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(300, 10)

    def forward(self, x):
        return self.linear(x)

class model_three(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(300, 10)
        self.bias = torch.zeros(10)

    def forward(self, x):
        return x @ self.weights + self.bias

if __name__ == '__main__':
    one = model_one()
    two = model_two()
    three = model_three()

    def show_parameters(model, model_name):
        print(model_name)
        for param in model.parameters():
            print(type(param.data), param.size())

    show_parameters(one, "model_one")
    show_parameters(two, "model_two")
    show_parameters(three, "model_three")

    # model_one
    # <class 'torch.Tensor'> torch.Size([300, 10])
    # <class 'torch.Tensor'> torch.Size([10])
    # model_two
    # <class 'torch.Tensor'> torch.Size([10, 300])
    # <class 'torch.Tensor'> torch.Size([10])
    # model_three

