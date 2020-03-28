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


