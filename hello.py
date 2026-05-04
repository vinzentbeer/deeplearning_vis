import torch 

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")
