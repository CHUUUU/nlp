import torch.nn.functional as F
import torch.nn as nn


class glu_custom(nn.Module):
    def __init__(self):
        super(glu_custom, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])