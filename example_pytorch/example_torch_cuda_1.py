import torch

# torch.cuda.synchronize()
    # CUDA 연산은 기본적으로 asynchronous하게 수행됩니다.
    # 이는 따로 synchronize하지 않는 한 CUDA 연산이 그 연산을 수행시킨 프로세스와 별개로 수행된다는 의미입니다.
    #
    # synchronize는 GPU 연산이 끝날 때까지 기다리는 함수입니다.
    # GPU 연산의 결과가 필요한 시점 이전에 synchronize를 사용하지 않으면
    # 딥러닝 결과의 올바름을 보장할 수 없습니다.

