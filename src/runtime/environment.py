"""
Environment metadata collection for experiment reproducibility.

Records: torch version, Python version, CUDA runtime, GPU model,
driver version, OS, backend, git SHA.
"""
import os
import platform
import subprocess
import sys
from typing import Any, Dict

import torch


def get_git_sha() -> str:
    """Get the current git commit SHA."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "unknown"


def collect_environment() -> Dict[str, Any]:
    """
    Collect comprehensive environment metadata for experiment reproducibility.

    Records:
      - torch.__version__
      - Python version
      - CUDA runtime version
      - GPU model
      - Driver version (where practical)
      - OS / platform
      - Backend
      - Git SHA
    """
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "git_sha": get_git_sha(),
    }

    # CUDA-specific information
    if torch.cuda.is_available():
        env["cuda_runtime_version"] = torch.version.cuda or "unknown"
        env["gpu_model"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()

        # Try to get driver version
        try:
            driver_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            env["driver_version"] = driver_output.split('\n')[0]
        except Exception:
            env["driver_version"] = "unknown"

        # cuDNN version
        if torch.backends.cudnn.is_available():
            env["cudnn_version"] = str(torch.backends.cudnn.version())
        else:
            env["cudnn_version"] = "unavailable"
    elif torch.backends.mps.is_available():
        env["gpu_model"] = "Apple Silicon MPS"
        env["cuda_runtime_version"] = "N/A"
        env["driver_version"] = "N/A"
    else:
        env["gpu_model"] = "CPU"
        env["cuda_runtime_version"] = "N/A"
        env["driver_version"] = "N/A"

    # Backend information (populated at distributed init time)
    try:
        import torch.distributed as dist_module
        if dist_module.is_initialized():
            env["backend"] = dist_module.get_backend()
        else:
            env["backend"] = "not_initialized"
    except Exception:
        env["backend"] = "unknown"

    return env
