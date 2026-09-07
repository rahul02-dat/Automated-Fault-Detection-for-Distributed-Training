import hashlib
from typing import Any
import torch

def compute_digest(state: Any) -> str:
    """
    Computes a stable SHA-256 digest of a canonicalized state.
    - Dicts: Keys are sorted and recursively hashed.
    - Lists/Tuples: Elements are recursively hashed in order.
    - Tensors: shape, dtype, and contiguous byte data are hashed.
    - Other types: String representation is hashed.
    """
    digest = hashlib.sha256()
    
    if isinstance(state, torch.Tensor):
        state_cpu = state.detach().cpu().contiguous()
        digest.update(str(state_cpu.dtype).encode('utf-8'))
        digest.update(str(state_cpu.shape).encode('utf-8'))
        digest.update(state_cpu.numpy().tobytes())
    elif isinstance(state, dict):
        for k in sorted(state.keys()):
            digest.update(str(k).encode('utf-8'))
            v_hash = compute_digest(state[k])
            digest.update(v_hash.encode('utf-8'))
    elif isinstance(state, (list, tuple)):
        for x in state:
            x_hash = compute_digest(x)
            digest.update(x_hash.encode('utf-8'))
    elif state is None:
        digest.update(b'null')
    else:
        digest.update(str(state).encode('utf-8'))
        
    return digest.hexdigest()
