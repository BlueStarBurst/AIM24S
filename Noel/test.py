import torch

tensor_cpu = torch.tensor([1, 2, 3])
tensor_cuda = None  # Initialize the tensor_cuda variable

# Check if CUDA is available
if torch.cuda.is_available():
    # If CUDA is available, move the tensor to the first GPU
    tensor_cuda = tensor_cpu.to(torch.device("cuda:0"))
    print("Tensor moved to GPU.")
else:
    print("CUDA is not available.")

# Verify that the tensor is now on the GPU
# print(tensor_cuda.device)
