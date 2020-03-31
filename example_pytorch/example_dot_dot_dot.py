import torch

a = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]])
print("a.shape : ", a.shape)

# b = a[...]
print(a[...].shape)
print(a[..., :].shape)
print(a[..., :, :].shape)
print(a[:, ...].shape)

print(a[..., :, :])

# a.shape :  torch.Size([2, 2, 3])
# torch.Size([2, 2, 3])
# torch.Size([2, 2, 3])
# torch.Size([2, 2, 3])
# torch.Size([2, 2, 3])

# 만약 a 가 10차원 이라면, 9, 10 번째 차원만 조작 하고 싶으면
# ... 으로 1~8 차원 까지 전체라는 말로 퉁 칠 수 있음