import subprocess

from knee_abnormality.autoresearch.runtime.remote import TrainingHost, remote

GPU_BUSY = 75
GPU_IDLE_MEMORY_MIB = 2048


def nvidia_smi(gpu: str, query: str, check: bool, host: dict | None):
    command = ["nvidia-smi", "-i", gpu, query, "--format=csv,noheader,nounits"]
    return subprocess.run(
        command if host is None else remote(TrainingHost(**host), command),
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


def gpu_available(gpu: str, host: dict | None = None):
    result = nvidia_smi(gpu, "--query-compute-apps=pid", False, host)
    if result.returncode:
        raise ValueError(f"Cannot inspect selected GPU: {result.stderr.strip()}")
    if result.stdout.strip():
        return False
    used = nvidia_smi(gpu, "--query-gpu=memory.used", True).stdout.strip()
    return int(used) < GPU_IDLE_MEMORY_MIB


def gpu_uuid(gpu: str, host: dict | None = None):
    value = nvidia_smi(gpu, "--query-gpu=uuid", True, host).stdout.strip()
    if not value.startswith("GPU-") or "\n" in value:
        raise ValueError("Select exactly one GPU of the training machine")
    return value
