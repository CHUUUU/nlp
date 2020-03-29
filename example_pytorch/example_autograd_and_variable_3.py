import torch
from torch.autograd import Variable

x = Variable(torch.ones(3), requires_grad=True)
y = x**2
z = y*3

print("x.data : ", x.data)
print("x.grad : ", x.grad)
print("x.grad_fn : ", x.grad_fn)

z.backward(torch.Tensor([0.1, 1, 10]))

print("#"*20)
print("x.data : ", x.data)
print("x.grad : ", x.grad)
print("x.grad_fn : ", x.grad_fn)


# x.data :  tensor([1., 1., 1.])
# x.grad :  None
# x.grad_fn :  None
# ####################
# x.data :  tensor([1., 1., 1.])
# x.grad :  tensor([ 0.6000,  6.0000, 60.0000])  <- torch.Tensor([0.1, 1, 10]) 과 곱해져서 반환
# x.grad_fn :  None
