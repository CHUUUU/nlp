## The Wrong Way

# Below timing method will NOT work for asynchronous cuda calls
import time as timer
start = timer.time()
loss.backward()
print("Time taken", timer.time() - start)  # tiny value



## The Right Way

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# whatever you are timing goes here
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))  # milliseconds





###

"""
Time the all_reduce_sum operation with model parameters size of the VGG-16 model (~138M floats)
    1. Paramters are loaded into each of the N=4 GPUs
    2. nccl.all_reduce is invoked on the paramters
TO get a breakdown of the VGG model size, see...
https://stackoverflow.com/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks
"""

from __future__ import print_function
import torch
import torch.cuda.nccl as nccl
import torch.cuda
import sys

model_size = 138000000  # of VGG-16
const_val = 5.0

nGPUs = torch.cuda.device_count()


def time_all_reduce_vgg_model_size(repeat=12, discard=2):
    print('repeat', repeat, 'discard', discard)
    times_per_iteration = []

    for _ in range(repeat):
        tensors = [torch.FloatTensor(model_size).fill_(const_val) for i in range(nGPUs)]  # dim size, value random
        expected = torch.FloatTensor(model_size).zero_()
        for tensor in tensors:
            expected.add_(tensor)  # add in-place on CPU

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]  # move ith tensor into ith GPU
        torch.cuda.synchronize()  # wait for move to complete

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        nccl.all_reduce(tensors)

        torch.cuda.synchronize()  # wait for all_reduce to complete
        end.record()

        torch.cuda.synchronize()  # need to wait once more for op to finish
        times_per_iteration.append(start.elapsed_time(end))  # millisecs

        for tensor in tensors:
            assert torch.all(torch.eq(tensor.cpu(), expected))  # move to CPU and compare

    times_per_iteration = times_per_iteration[discard:]  # discard first few
    print(len(times_per_iteration), times_per_iteration)
    avg_time_taken = sum(times_per_iteration)/(repeat - discard)
    return avg_time_taken

reduction_time = time_all_reduce_vgg_model_size(12, 2)

print('Python VERSION:', sys.version)
print('PyTorch VERSION:', torch.__version__)
print('RCCL VERSION:', torch.cuda.nccl.version())
print ('Available GPUs ', nGPUs)
print("Time to all_reduce {} float tensors:{:.6f} milliseconds".format(model_size, reduction_time))
print("Ring reduce rate:{:.6} GB/s".format((model_size * 4 * 2 * (nGPUs -1) / nGPUs) / (reduction_time /1000) /1.0e9))