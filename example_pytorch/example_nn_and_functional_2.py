import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
filter = Variable(torch.ones(1, 1, 3, 3))
# Functional 을 사용하기 위해선 wight 가 따로 필요, Filter 는 conv2d 의 weight
# 미분 계산 되어지기 위해, grad 와 grad_fn 이 있는 Variable 로 형태 변환
# requires_grad 를 선언해주어야지 backward 가 가능

out = F.conv2d(input, filter)
out.backward()
print("out : ", out)

print("out.grad_fn : ", out.grad_fn)
print("out.grad : ", out.grad)

print("input.grad_fn : ", input.grad_fn)
print("input.grad : ", input.grad)

# out :  tensor([[[[9.]]]], grad_fn=<ThnnConv2DBackward>)  <- bias 가 없어 그대로 나옴
# out.grad_fn :  <ThnnConv2DBackward object at 0x000001E02FF46198>
# out.grad :  None
# input.grad_fn :  None
# input.grad :  tensor([[[[1., 1., 1.],
#                           [1., 1., 1.],
#                           [1., 1., 1.]]]])