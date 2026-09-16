import os


def configure_cuda_allocator() -> str:
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if name in os.environ:
            return os.environ[name]
    settings = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = settings
    return settings
