import torch
print(torch.cuda.is_available())  # True면 GPU 사용 가능
print(torch.version.cuda)         # CUDA 버전
