"""
Deterministic seed management and RNG state capture/restore.

Seed policy:
  global_seed = S
  rank_seed = S + rank

Rank offset policy:
  Each rank derives its seed by adding its rank to the global seed.
  This ensures different but deterministic data sampling per rank.

Checkpoint restoration policy:
  RNG state is captured as a dict of {source: state} and can be
  restored from a checkpoint to ensure bit-exact resumption under
  deterministic conditions.
"""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_deterministic_seeds(global_seed: int, rank: int, local_rank: Optional[int] = None):
    """
    Establish deterministic seeds for the experiment.

    global_seed = S
    rank_seed = S + rank
    """
    rank_seed = global_seed + rank

    # Python RNG
    random.seed(rank_seed)

    # NumPy RNG
    np.random.seed(rank_seed)

    # PyTorch CPU RNG
    torch.manual_seed(rank_seed)

    # PyTorch CUDA RNG if applicable
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)

    # Set deterministic algorithms if required for absolute strictness,
    # though this can hurt performance or crash on some operations.
    # We will leave these commented out unless strictly needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def capture_rng_state(rank: int = 0) -> Dict[str, Any]:
    """
    Capture the current RNG state from all sources.

    Returns a dict with keys:
      - torch_cpu: PyTorch CPU RNG state (ByteTensor)
      - python: Python random module state (tuple)
      - numpy: NumPy RNG state (dict)
      - torch_cuda_<device>: PyTorch CUDA RNG state per device (if available)

    Each entry is the full state object needed for restore_rng_state().
    """
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            state[f"torch_cuda_{i}"] = torch.cuda.get_rng_state(i)

    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG state from a previously captured dict.

    This must be called after checkpoint restore to ensure bit-exact
    continuation of the random number sequence.
    """
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if torch.cuda.is_available():
        for key, val in state.items():
            if key.startswith("torch_cuda_"):
                device_idx = int(key.split("_")[-1])
                torch.cuda.set_rng_state(val, device_idx)
