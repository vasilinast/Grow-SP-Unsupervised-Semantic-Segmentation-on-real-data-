import torch

# Verify CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

# Test a simple CUDA tensor operation
device = torch.device('cuda')
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)
c = a + b

print("CUDA operation result:", c)  # Should print the tensor [5.0, 7.0, 9.0] on the GPU
print("CUDA is available. GPU will be used.")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA current device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(0))
print(torch.__version__)
print(torch.version.cuda)
