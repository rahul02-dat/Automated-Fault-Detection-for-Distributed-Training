import collections
from typing import Any, Dict, Optional

import torch

from .contracts import Comparator
from .errors import UnsupportedComparatorError
from .hashing import compute_digest

def canonicalize(state: Any) -> Any:
    """
    Canonicalize state before hashing or structured comparison.
    - Moves tensors to CPU and detaches them.
    - Sorts dictionary keys deterministically.
    - Preserves lists/tuples.
    """
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    elif isinstance(state, dict):
        return {k: canonicalize(v) for k, v in sorted(state.items())}
    elif isinstance(state, list):
        return [canonicalize(x) for x in state]
    elif isinstance(state, tuple):
        return tuple(canonicalize(x) for x in state)
    else:
        return state

def compare_exact(a: Any, b: Any) -> Dict[str, Any]:
    """Strict equality comparison."""
    if type(a) != type(b):
        return {"match": False, "reason": f"Type mismatch: {type(a).__name__} != {type(b).__name__}"}
        
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return {"match": False, "reason": f"Shape mismatch: {a.shape} != {b.shape}"}
        if a.dtype != b.dtype:
            return {"match": False, "reason": f"Dtype mismatch: {a.dtype} != {b.dtype}"}
        if not torch.equal(a, b):
            diff = torch.abs(a - b)
            return {
                "match": False, 
                "reason": "Tensor values mismatch",
                "max_abs_diff": torch.max(diff).item(),
                "mean_abs_diff": torch.mean(diff.float()).item()
            }
        return {"match": True}
        
    if isinstance(a, dict):
        if a.keys() != b.keys():
            missing = b.keys() - a.keys()
            extra = a.keys() - b.keys()
            return {"match": False, "reason": f"Dict keys mismatch. Missing: {missing}, Extra: {extra}"}
        for k in a:
            res = compare_exact(a[k], b[k])
            if not res["match"]:
                res["reason"] = f"Key '{k}' mismatch: {res['reason']}"
                return res
        return {"match": True}
        
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {"match": False, "reason": f"Length mismatch: {len(a)} != {len(b)}"}
        for i, (va, vb) in enumerate(zip(a, b)):
            res = compare_exact(va, vb)
            if not res["match"]:
                res["reason"] = f"Index {i} mismatch: {res['reason']}"
                return res
        return {"match": True}
        
    if a != b:
        return {"match": False, "reason": f"Value mismatch: {a} != {b}"}
    return {"match": True}


def compare_allclose(a: Any, b: Any, rtol: float, atol: float) -> Dict[str, Any]:
    """Floating point tolerance equality."""
    if type(a) != type(b):
        return {"match": False, "reason": f"Type mismatch: {type(a).__name__} != {type(b).__name__}"}
        
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return {"match": False, "reason": f"Shape mismatch: {a.shape} != {b.shape}"}
        if a.dtype != b.dtype:
            return {"match": False, "reason": f"Dtype mismatch: {a.dtype} != {b.dtype}"}
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = torch.abs(a - b)
            # Safe rel_diff
            rel_diff = diff / (torch.abs(b) + 1e-8)
            return {
                "match": False, 
                "reason": "Tensor allclose mismatch",
                "max_abs_diff": torch.max(diff).item(),
                "mean_abs_diff": torch.mean(diff.float()).item(),
                "max_rel_diff": torch.max(rel_diff).item()
            }
        return {"match": True}
        
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {"match": False, "reason": "Dict keys mismatch"}
        for k in a:
            res = compare_allclose(a[k], b[k], rtol, atol)
            if not res["match"]:
                res["reason"] = f"Key '{k}' mismatch: {res['reason']}"
                return res
        return {"match": True}
        
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {"match": False, "reason": "Length mismatch"}
        for i, (va, vb) in enumerate(zip(a, b)):
            res = compare_allclose(va, vb, rtol, atol)
            if not res["match"]:
                res["reason"] = f"Index {i} mismatch: {res['reason']}"
                return res
        return {"match": True}
        
    # Fallback to exact for non-tensors
    return compare_exact(a, b)


def compare_hash(a: Any, b: Any) -> Dict[str, Any]:
    """Hash digest equality using SHA-256."""
    try:
        hash_a = compute_digest(a)
        hash_b = compute_digest(b)
        if hash_a != hash_b:
            return {"match": False, "reason": "Hash mismatch", "hash_expected": hash_a, "hash_actual": hash_b}
        return {"match": True}
    except Exception as e:
        return {"match": False, "reason": f"Hashing failed: {e}"}


def compare_state(
    a: Any, b: Any, comparator: Comparator, rtol: Optional[float] = None, atol: Optional[float] = None
) -> Dict[str, Any]:
    """Compare two canonicalized states using the specified comparator."""
    if comparator == Comparator.EXACT:
        return compare_exact(a, b)
    elif comparator == Comparator.ALLCLOSE:
        if rtol is None or atol is None:
            raise ValueError("ALLCLOSE requires rtol and atol")
        return compare_allclose(a, b, rtol=rtol, atol=atol)
    elif comparator == Comparator.HASH:
        return compare_hash(a, b)
    elif comparator in (Comparator.NORM, Comparator.CUSTOM):
        raise UnsupportedComparatorError(f"Comparator {comparator.name} is not fully implemented yet.")
    
    raise UnsupportedComparatorError(f"Unsupported comparator: {comparator}")
