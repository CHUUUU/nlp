import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
filter = Variable(torch.ones(1, 1, 3, 3))

with torch.no_grad():
    out = F.conv2d(input, filter)
    out.backward()
    print("out : ", out)

    print("out.grad_fn : ", out.grad_fn)
    print("out.grad : ", out.grad)

    print("input.grad_fn : ", input.grad_fn)
    print("input.grad : ", input.grad)

# Traceback (most recent call last):
#   File "test.py", line 11, in <module>
#     out.backward()
#   File "E:\MYGIT~1\venv\lib\site-packages\torch\tensor.py", line 195, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph)
#   File "E:\MYGIT~1\venv\lib\site-packages\torch\autograd\__init__.py", line 99, in backward
#     allow_unreachable=True)  # allow_unreachable flag
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
