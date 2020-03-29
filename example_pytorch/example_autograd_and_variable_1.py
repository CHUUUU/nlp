import torch
from torch.autograd import Variable

a = Variable(torch.ones(2,2))
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

# a.data :  tensor([[1., 1.],
#         [1., 1.]])
# a.grad :  None
# a.grad_fn :  None
# Traceback (most recent call last):
#   File "test.py", line 12, in <module>
#     out.backward()
#   File "E:\MYGIT~1\venv\lib\site-packages\torch\tensor.py", line 195, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph)
#   File "E:\MYGIT~1\venv\lib\site-packages\torch\autograd\__init__.py", line 99, in backward
#     allow_unreachable=True)  # allow_unreachable flag
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
