import torch
from torch.autograd import Variable

a = Variable(torch.ones(2,2), requires_grad=True)
b = a + 2
c = b**2
with torch.no_grad():
    d = b + 2
    print(d)
out = c.sum()
out.backward()


# tensor([[5., 5.],
#         [5., 5.]])
