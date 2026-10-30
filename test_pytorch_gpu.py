
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = a @ b

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Output shape:", c.shape)
print("GPU computation successful:", c.is_cuda)
