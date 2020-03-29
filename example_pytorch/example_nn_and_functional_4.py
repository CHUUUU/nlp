import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)

func = nn.Conv2d(1, 1, 3, bias=False)
print("func.weight : ", func.weight)
print("#"*10)

# func.weight 에 따로 값 지정
func.weight = torch.nn.Parameter(torch.ones(1, 1, 3, 3) + 1)
print("func.weight : ", func.weight)
print("#"*10)

out = func(input)
out.backward()
print("out : ", out)

print("out.grad_fn : ", out.grad_fn)
print("out.grad : ", out.grad)

print("input.grad_fn : ", input.grad_fn)
print("input.grad : ", input.grad)


# func.weight :  Parameter containing:
# tensor([[[[-0.3235,  0.2274, -0.0551],
#           [-0.1103, -0.2733, -0.0588],
#           [ 0.1591,  0.1861,  0.1848]]]], requires_grad=True)
# ##########
# func.weight :  Parameter containing:
# tensor([[[[2., 2., 2.],
#           [2., 2., 2.],
#           [2., 2., 2.]]]], requires_grad=True)
# ##########
# out :  tensor([[[[18.]]]], grad_fn=<ThnnConv2DBackward>)
# out.grad_fn :  <ThnnConv2DBackward object at 0x0000018E22FA20B8>
# out.grad :  None
# input.grad_fn :  None
# input.grad :  tensor([[[[2., 2., 2.],
#                           [2., 2., 2.],
#                           [2., 2., 2.]]]])