import platform
import subprocess
import torch
from typing import Dict, Any

def get_git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def collect_environment() -> Dict[str, Any]:
    """Collect deterministic environment metadata."""
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "git_sha": get_git_sha(),
    }
    
    if torch.cuda.is_available():
        env["gpu_model"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        env["gpu_model"] = "Apple Silicon MPS"
    else:
        env["gpu_model"] = "CPU"
        
    return env
