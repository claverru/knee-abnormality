import os

import torch


def configure_cuda_allocator() -> str:
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if name in os.environ:
            return os.environ[name]
    settings = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = settings
    return settings


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
