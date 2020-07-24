# torch.cuda는 CUDA 작업을 설정하고 실행하는 데 사용됩니다. 
# 현재 선택된 GPU를 추적하고 할당 한 모든 CUDA 텐서는 기본적으로 해당 장치에 생성됩니다. 
# 선택한 장치는 torch.cuda.device 컨텍스트 관리자를 사용하여 변경할 수 있습니다.

# 그러나 텐서가 할당되면 선택한 장치에 관계없이 텐서에서 작업을 수행 할 수 있으며 
# 결과는 항상 텐서와 동일한 장치에 배치됩니다.

# copy_ () 및 to () 및 cuda ()와 같은 복사와 같은 기능을 가진 다른 메소드를 제외하고 
# 기본적으로 크로스 GPU 작업은 허용되지 않습니다. 
# 피어 투 피어 메모리 액세스를 활성화하지 않으면 
# 다른 장치에 분산된 텐서에서 
# op를 시작하려고하면 오류가 발생합니다.


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
