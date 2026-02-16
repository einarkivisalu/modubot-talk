import torch
import time

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

device = "cuda"

# Small benchmark function
def benchmark(dtype, label):
    x = torch.randn(4096, 4096, device=device, dtype=dtype)
    y = torch.randn(4096, 4096, device=device, dtype=dtype)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(20):
        z = x @ y

    torch.cuda.synchronize()
    print(f"{label} time:", time.time() - t0, "seconds")


# Run FP32 test
benchmark(torch.float32, "FP32")

# Run BF16 test (only if supported)
if torch.cuda.is_bf16_supported():
    benchmark(torch.bfloat16, "BF16")
