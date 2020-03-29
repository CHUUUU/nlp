import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
# nn 을 사용하기 위해선 wight 가 따로 필요없음
# 대신 함수를 선언해 주어야 함
# 미분 계산 되어지기 위해, grad 와 grad_fn 이 있는 Variable 로 형태 변환
# requires_grad 를 선언해주어야지 backward 가 가능

func = nn.Conv2d(1, 1, 3)
print(func.weight)  # func 에 (1, 1, 3)에 대한 weight 를 가지고 있음
# nn 은 자동으로 bias 를 가지고 있고 True 이다.
out = func(input)
out.backward()
print("out : ", out)

print("out.grad_fn : ", out.grad_fn)
print("out.grad : ", out.grad)

print("input.grad_fn : ", input.grad_fn)
print("input.grad : ", input.grad)

# Parameter containing:
# tensor([[[[ 0.1751, -0.0854, -0.2196],
#           [-0.0453, -0.0542, -0.2257],
#           [ 0.2035, -0.2871,  0.0548]]]], requires_grad=True)
# out :  tensor([[[[-0.4232]]]], grad_fn=<ThnnConv2DBackward>) <- bias 가 들어가서 F 와 값이 다름
# out.grad_fn :  <ThnnConv2DBackward object at 0x00000134982EF208>
# out.grad :  None
# input.grad_fn :  None
# input.grad :  tensor([[[[-0.1544,  0.0974,  0.1198],
#                           [-0.1294,  0.2110, -0.0968],
#                           [ 0.1233,  0.0678,  0.2074]]]])
