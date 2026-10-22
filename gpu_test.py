
import torch

print("PyTorch version:", torch.__version__)
print("XPU support available:", hasattr(torch, "xpu"))

if hasattr(torch, "xpu"):
    print("Intel GPU detected:", torch.xpu.is_available())

    if torch.xpu.is_available():
        print("GPU:", torch.xpu.get_device_name(0))

        x = torch.rand(1000, 1000, device="xpu")
        y = x @ x
        print("GPU computation successful:", y.shape)
else:
    print("This PyTorch build does not expose XPU support.")
