import torch

# We will use GPU-0 and GPU-1.
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)

to_copy = torch.ones(100, device=0)
to_calc = torch.rand(100, device=0)

# Introduce a separate stream for copy and synchronize to the default stream.
# The copy will be started when "to_copy" is ready.
default_stream = torch.cuda.default_stream(0)
copy_stream = torch.cuda.Stream(0)
copy_stream.wait_stream(default_stream)

with torch.cuda.stream(copy_stream):
    # Both the copy and computation on "to_calc" will start at the same time.
    # But the copy will be finished later.
    torch.cuda._sleep(100000000)
    copied = to_copy.to(1)

# Here's any computation which allocates some new tensors on the default stream.
to_calc * 100

# Free "to_copy".
default_stream.wait_stream(copy_stream)
del to_copy

to_calc * 200

print(copied.sum().item())