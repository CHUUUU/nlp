############################
## The Wrong Way
############################
# Below timing method will NOT work for asynchronous cuda calls
import time as timer
start = timer.time()
loss.backward()
print("Time taken", timer.time() - start)  # tiny value


#############################
## The Right Way
#############################
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# whatever you are timing goes here
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))  # milliseconds



#####################
## torch.cuda.Event
#####################
# CUDA events are 
#     - synchronization markers that can be used to monitor the device’s progress, 
#     - to accurately measure timing, and to synchronize CUDA streams.
# The underlying CUDA events are 
#     - lazily initialized when the event is first recorded or exported to another process. 
# After creation, 
#     - only streams on the same device may record the event. 
# However, 
#     - streams on any device can wait on the event.

# Parameters
#     - enable_timing (bool, optional) – indicates if the event should measure time (default: False)
#     - blocking (bool, optional) – if True, wait() will be blocking (default: False)
#     - interprocess (bool) – if True, the event can be shared between processes (default: False)




###################
## example
###################
## https://github.com/idiap/fast-transformers

# softmax_start = torch.cuda.Event(enable_timing=True)
# softmax_end = torch.cuda.Event(enable_timing=True)
# linear_start = torch.cuda.Event(enable_timing=True)
# linear_end = torch.cuda.Event(enable_timing=True)

# with torch.no_grad():
#     softmax_start.record()
#     y = softmax_model(X)
#     softmax_end.record()
#     torch.cuda.synchronize()
#     print("Softmax: ", softmax_start.elapsed_time(softmax_end), "ms")
#     # Softmax: 144 ms (on a GTX1080Ti)

# with torch.no_grad():
#     linear_start.record()
#     y = linear_model(X)
#     linear_end.record()
#     torch.cuda.synchronize()
#     print("Linear: ", linear_start.elapsed_time(linear_end), "ms")
#     # Linear: 68 ms (on a GTX1080Ti)
