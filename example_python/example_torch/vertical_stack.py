import torch

a = torch.arange(1, 10)
b = a[..., None]

print(a)
print(b)
print(a[None, ...])
print(b[None, ...])


# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6],
#         [7],
#         [8],
#         [9]])
# tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
# tensor([[[1],
#          [2],
#          [3],
#          [4],
#          [5],
#          [6],
#          [7],
#          [8],
#          [9]]])
