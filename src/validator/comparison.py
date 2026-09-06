import collections
from typing import Any, Dict, Tuple, Optional

import torch

from .contracts import Comparator
from .errors import UnsupportedComparatorError


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


def compare_exact(a: Any, b: Any) -> Tuple[bool, Optional[str]]:
    """Strict equality comparison."""
    if type(a) != type(b):
        return False, f"Type mismatch: {type(a).__name__} != {type(b).__name__}"
        
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return False, f"Shape mismatch: {a.shape} != {b.shape}"
        if a.dtype != b.dtype:
            return False, f"Dtype mismatch: {a.dtype} != {b.dtype}"
        if not torch.equal(a, b):
            return False, "Tensor values mismatch"
        return True, None
        
    if isinstance(a, dict):
        if a.keys() != b.keys():
            missing = b.keys() - a.keys()
            extra = a.keys() - b.keys()
            return False, f"Dict keys mismatch. Missing: {missing}, Extra: {extra}"
        for k in a:
            match, reason = compare_exact(a[k], b[k])
            if not match:
                return False, f"Key '{k}' mismatch: {reason}"
        return True, None
        
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f"Length mismatch: {len(a)} != {len(b)}"
        for i, (va, vb) in enumerate(zip(a, b)):
            match, reason = compare_exact(va, vb)
            if not match:
                return False, f"Index {i} mismatch: {reason}"
        return True, None
        
    if a != b:
        return False, f"Value mismatch: {a} != {b}"
    return True, None


def compare_allclose(a: Any, b: Any, rtol: float, atol: float) -> Tuple[bool, Optional[str]]:
    """Floating point tolerance equality."""
    if type(a) != type(b):
        return False, f"Type mismatch: {type(a).__name__} != {type(b).__name__}"
        
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return False, f"Shape mismatch: {a.shape} != {b.shape}"
        if a.dtype != b.dtype:
            return False, f"Dtype mismatch: {a.dtype} != {b.dtype}"
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            max_abs = torch.max(torch.abs(a - b)).item()
            return False, f"Tensor allclose mismatch. Max abs diff: {max_abs}"
        return True, None
        
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False, "Dict keys mismatch"
        for k in a:
            match, reason = compare_allclose(a[k], b[k], rtol, atol)
            if not match:
                return False, f"Key '{k}' mismatch: {reason}"
        return True, None
        
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, "Length mismatch"
        for i, (va, vb) in enumerate(zip(a, b)):
            match, reason = compare_allclose(va, vb, rtol, atol)
            if not match:
                return False, f"Index {i} mismatch: {reason}"
        return True, None
        
    # Fallback to exact for non-tensors
    return compare_exact(a, b)


def compare_hash(a: Any, b: Any) -> Tuple[bool, Optional[str]]:
    """Hash digest equality (placeholder)."""
    # Simple Python hash for now, structured hashing would be better for nested.
    # To properly implement this, we'd need a robust recursive hasher.
    try:
        if hash(str(a)) != hash(str(b)):
            return False, "Hash mismatch"
        return True, None
    except Exception as e:
        return False, f"Hashing failed: {e}"


def compare_state(
    a: Any, b: Any, comparator: Comparator, rtol: Optional[float] = None, atol: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """Compare two canonicalized states using the specified comparator."""
    if comparator == Comparator.EXACT:
        return compare_exact(a, b)
    elif comparator == Comparator.ALLCLOSE:
        if rtol is None or atol is None:
            raise ValueError("ALLCLOSE requires rtol and atol")
        return compare_allclose(a, b, rtol=rtol, atol=atol)
    elif comparator == Comparator.HASH:
        return compare_hash(a, b)
    elif comparator == Comparator.NORM:
        # Simplistic norm compare, useful for diagnostics
        return False, "NORM comparator not fully implemented"
    elif comparator == Comparator.CUSTOM:
        return False, "CUSTOM comparator requires custom logic"
    
    raise UnsupportedComparatorError(f"Unsupported comparator: {comparator}")
