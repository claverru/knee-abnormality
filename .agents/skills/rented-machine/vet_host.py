import multiprocessing
import subprocess
import time

import numpy as np
import timm
import torch


def timed(function, repeats):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def copy_rate(seconds):
    source = np.ones(64 * 2**20, dtype=np.uint8)
    end = time.perf_counter() + seconds
    count = 0
    while time.perf_counter() < end:
        source.copy()
        count += 1
    return count * 64 / 1024 / seconds


def gpu() -> None:
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    print(f"bf16 matmul: {2 * 8192**3 / timed(lambda: a @ a, 20) / 1e12:.0f} TFLOPS")
    x = torch.empty(512 * 2**20, dtype=torch.uint8, device=device)
    y = torch.empty_like(x)
    print(f"device copy: {2 * x.numel() / timed(lambda: y.copy_(x), 50) / 1e9:.0f} GB/s")
    host = torch.empty(512 * 2**20, dtype=torch.uint8).pin_memory()
    seconds = timed(lambda: x.copy_(host, non_blocking=True), 20)
    print(f"host to device: {host.numel() / seconds / 1e9:.1f} GB/s")
    start = time.perf_counter()
    for _ in range(20000):
        torch.empty(1, device=device).add_(1)
    torch.cuda.synchronize()
    print(f"small-kernel launch: {(time.perf_counter() - start) / 20000 * 1e6:.1f} us")
    model = timm.create_model("convnext_tiny", pretrained=False).to(device).eval()
    model = model.to(memory_format=torch.channels_last)
    batch = torch.randn(128, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    query = "power.draw,power.limit,clocks.sm,temperature.gpu,clocks_event_reasons.sw_power_cap"
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while time.perf_counter() - start < 60:
            tick = time.perf_counter()
            for _ in range(20):
                model(batch)
            torch.cuda.synchronize()
            rate = 20 * len(batch) / (time.perf_counter() - tick)
            state = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
            print(f"t={time.perf_counter() - start:4.0f}s {rate:6.0f} img/s | {state}", flush=True)


def cpu() -> None:
    for processes in (1, 4, 8, 16, 28):
        with multiprocessing.Pool(processes) as pool:
            total = sum(pool.map(copy_rate, [4.0] * processes))
        print(f"memory copy, {processes:2d} processes: {total:5.1f} GB/s", flush=True)


if __name__ == "__main__":
    gpu()
    cpu()
