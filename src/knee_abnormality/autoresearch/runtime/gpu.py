import subprocess

GPU_BUSY = 75


def nvidia_smi(gpu: str, query: str, check: bool):
    return subprocess.run(
        ["nvidia-smi", "-i", gpu, query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=check,
    )


def gpu_available(gpu: str):
    result = nvidia_smi(gpu, "--query-compute-apps=pid", False)
    if result.returncode:
        raise ValueError(f"Cannot inspect selected GPU: {result.stderr.strip()}")
    return not result.stdout.strip()


def gpu_uuid(gpu: str):
    value = nvidia_smi(gpu, "--query-gpu=uuid", True).stdout.strip()
    if not value.startswith("GPU-") or "\n" in value:
        raise ValueError("Select exactly one local GPU")
    return value
