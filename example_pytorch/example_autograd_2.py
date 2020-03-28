import torch
from torch.autograd import Variable

a = Variable(torch.ones(2,2), requires_grad=True)
print("a.data : ", a.data)
print("a.grad : ", a.grad)
print("a.grad_fn : ", a.grad_fn)

b = a + 2
c = b**2
out = c.sum()
out.backward()

print("#"*20)
print("a.data : ", a.data)
print("a.grad : ", a.grad)
print("a.grad_fn : ", a.grad_fn)

print("#"*20)
print("b.data : ", b.data)
print("b.grad : ", b.grad)
print("b.grad_fn : ", b.grad_fn)


# a.data :  tensor([[1., 1.],
#                   [1., 1.]])
# a.grad :  None
# a.grad_fn :  None
# ####################
# a.data :  tensor([[1., 1.],
#                   [1., 1.]])
# a.grad :  tensor([[6., 6.],
#                   [6., 6.]])
# a.grad_fn :  None

# a.grad = ∂out/∂a  (out.backward() 로 인해서 미분 값이 들어감)
# a.grad_fn = a 가 직접적으로 수행한 연산이 없기 때문에, 함수가 없음

# ####################
# b.data :  tensor([[3., 3.],
#                   [3., 3.]])
# b.grad :  None
# b.grad_fn :  <AddBackward0 object at 0x0000013C7D72C4A8>

# b.grad : a.grad 에 대한 것이기에 (variable), b의 grad 는 필요 없다.
# b.grad_fn : 대신 b는 a에 대해 + 연산을 해준다 (그래서 add 에 대한 backward 가 찍힘)

