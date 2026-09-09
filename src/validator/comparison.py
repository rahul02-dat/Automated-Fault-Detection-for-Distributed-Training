"""
State comparison functions for cross-rank validation.

Supports EXACT, ALLCLOSE, and HASH comparators.
All comparison functions return rich diagnostics including key paths,
shape, dtype, and quantitative differences.
"""
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


def compare_exact(a: Any, b: Any, _key_path: str = "") -> Dict[str, Any]:
    """Strict equality comparison with key-path tracking."""
    if type(a) != type(b):
        return {
            "match": False,
            "reason": f"Type mismatch: {type(a).__name__} != {type(b).__name__}",
            "key_path": _key_path,
            "type_a": type(a).__name__,
            "type_b": type(b).__name__,
        }

    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return {
                "match": False,
                "reason": f"Shape mismatch: {a.shape} != {b.shape}",
                "key_path": _key_path,
                "shape_a": str(tuple(a.shape)),
                "shape_b": str(tuple(b.shape)),
            }
        if a.dtype != b.dtype:
            return {
                "match": False,
                "reason": f"Dtype mismatch: {a.dtype} != {b.dtype}",
                "key_path": _key_path,
                "dtype_a": str(a.dtype),
                "dtype_b": str(b.dtype),
            }
        # Handle NaN: NaN != NaN, so check explicitly
        a_nan = torch.isnan(a)
        b_nan = torch.isnan(b)
        if a_nan.any() or b_nan.any():
            if not torch.equal(a_nan, b_nan):
                return {
                    "match": False,
                    "reason": "NaN location mismatch",
                    "key_path": _key_path,
                    "nan_count_a": a_nan.sum().item(),
                    "nan_count_b": b_nan.sum().item(),
                }
        # Handle Inf
        a_inf = torch.isinf(a)
        b_inf = torch.isinf(b)
        if a_inf.any() or b_inf.any():
            if not torch.equal(a_inf, b_inf):
                return {
                    "match": False,
                    "reason": "Inf location mismatch",
                    "key_path": _key_path,
                }

        if not torch.equal(a, b):
            diff = torch.abs(a.float() - b.float())
            # First mismatch index
            ne_mask = (a != b)
            first_idx = None
            if ne_mask.any():
                idx = ne_mask.nonzero(as_tuple=False)[0]
                first_idx = str(tuple(idx.tolist()))
            # Safe relative difference
            rel_diff = diff / (torch.abs(b.float()) + 1e-8)
            return {
                "match": False,
                "reason": "Tensor values mismatch",
                "key_path": _key_path,
                "shape": str(tuple(a.shape)),
                "dtype": str(a.dtype),
                "max_abs_diff": torch.max(diff).item(),
                "mean_abs_diff": torch.mean(diff).item(),
                "max_rel_diff": torch.max(rel_diff).item(),
                "first_mismatch_index": first_idx,
            }
        return {"match": True, "key_path": _key_path}

    if isinstance(a, dict):
        if a.keys() != b.keys():
            missing = b.keys() - a.keys()
            extra = a.keys() - b.keys()
            return {
                "match": False,
                "reason": f"Dict keys mismatch. Missing: {missing}, Extra: {extra}",
                "key_path": _key_path,
            }
        for k in sorted(a.keys()):
            child_path = f"{_key_path}.{k}" if _key_path else str(k)
            res = compare_exact(a[k], b[k], _key_path=child_path)
            if not res["match"]:
                return res
        return {"match": True, "key_path": _key_path}

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {
                "match": False,
                "reason": f"Length mismatch: {len(a)} != {len(b)}",
                "key_path": _key_path,
            }
        for i, (va, vb) in enumerate(zip(a, b)):
            child_path = f"{_key_path}[{i}]" if _key_path else f"[{i}]"
            res = compare_exact(va, vb, _key_path=child_path)
            if not res["match"]:
                return res
        return {"match": True, "key_path": _key_path}

    if a != b:
        return {
            "match": False,
            "reason": f"Value mismatch: {a} != {b}",
            "key_path": _key_path,
        }
    return {"match": True, "key_path": _key_path}


def compare_allclose(a: Any, b: Any, rtol: float, atol: float, _key_path: str = "") -> Dict[str, Any]:
    """Floating point tolerance equality with key-path tracking."""
    if type(a) != type(b):
        return {
            "match": False,
            "reason": f"Type mismatch: {type(a).__name__} != {type(b).__name__}",
            "key_path": _key_path,
        }

    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return {
                "match": False,
                "reason": f"Shape mismatch: {a.shape} != {b.shape}",
                "key_path": _key_path,
                "shape_a": str(tuple(a.shape)),
                "shape_b": str(tuple(b.shape)),
            }
        if a.dtype != b.dtype:
            return {
                "match": False,
                "reason": f"Dtype mismatch: {a.dtype} != {b.dtype}",
                "key_path": _key_path,
                "dtype_a": str(a.dtype),
                "dtype_b": str(b.dtype),
            }
        # Handle NaN/Inf
        a_nan = torch.isnan(a)
        b_nan = torch.isnan(b)
        if a_nan.any() or b_nan.any():
            if not torch.equal(a_nan, b_nan):
                return {
                    "match": False,
                    "reason": "NaN location mismatch",
                    "key_path": _key_path,
                }

        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = torch.abs(a.float() - b.float())
            # Safe rel_diff
            rel_diff = diff / (torch.abs(b.float()) + 1e-8)
            # First mismatch beyond tolerance
            exceeds = diff > (atol + rtol * torch.abs(b.float()))
            first_idx = None
            if exceeds.any():
                idx = exceeds.nonzero(as_tuple=False)[0]
                first_idx = str(tuple(idx.tolist()))
            return {
                "match": False,
                "reason": "Tensor allclose mismatch",
                "key_path": _key_path,
                "shape": str(tuple(a.shape)),
                "dtype": str(a.dtype),
                "rtol": rtol,
                "atol": atol,
                "max_abs_diff": torch.max(diff).item(),
                "mean_abs_diff": torch.mean(diff).item(),
                "max_rel_diff": torch.max(rel_diff).item(),
                "first_mismatch_index": first_idx,
            }
        return {"match": True, "key_path": _key_path}

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {
                "match": False,
                "reason": "Dict keys mismatch",
                "key_path": _key_path,
            }
        for k in sorted(a.keys()):
            child_path = f"{_key_path}.{k}" if _key_path else str(k)
            res = compare_allclose(a[k], b[k], rtol, atol, _key_path=child_path)
            if not res["match"]:
                return res
        return {"match": True, "key_path": _key_path}

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {
                "match": False,
                "reason": "Length mismatch",
                "key_path": _key_path,
            }
        for i, (va, vb) in enumerate(zip(a, b)):
            child_path = f"{_key_path}[{i}]" if _key_path else f"[{i}]"
            res = compare_allclose(va, vb, rtol, atol, _key_path=child_path)
            if not res["match"]:
                return res
        return {"match": True, "key_path": _key_path}

    # Fallback to exact for non-tensors
    return compare_exact(a, b, _key_path=_key_path)


def compare_hash(a: Any, b: Any, _key_path: str = "") -> Dict[str, Any]:
    """Hash digest equality using SHA-256."""
    try:
        hash_a = compute_digest(a)
        hash_b = compute_digest(b)
        if hash_a != hash_b:
            return {
                "match": False,
                "reason": "Hash mismatch",
                "key_path": _key_path,
                "hash_expected": hash_a,
                "hash_actual": hash_b,
            }
        return {"match": True, "key_path": _key_path}
    except Exception as e:
        return {
            "match": False,
            "reason": f"Hashing failed: {e}",
            "key_path": _key_path,
        }


def compare_state(
    a: Any, b: Any, comparator: Comparator, rtol: Optional[float] = None, atol: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compare two canonicalized states using the specified comparator.

    Supported: EXACT, ALLCLOSE, HASH.
    Unsupported (NORM, CUSTOM) will raise UnsupportedComparatorError.
    """
    result: Dict[str, Any]

    if comparator == Comparator.EXACT:
        result = compare_exact(a, b)
    elif comparator == Comparator.ALLCLOSE:
        if rtol is None or atol is None:
            raise ValueError("ALLCLOSE requires rtol and atol")
        result = compare_allclose(a, b, rtol=rtol, atol=atol)
    elif comparator == Comparator.HASH:
        result = compare_hash(a, b)
    elif comparator in (Comparator.NORM, Comparator.CUSTOM):
        raise UnsupportedComparatorError(
            f"Comparator {comparator.value} is not implemented. "
            f"Supported comparators: EXACT, ALLCLOSE, HASH. "
            f"This configuration error should be caught at contract registration time."
        )
    else:
        raise UnsupportedComparatorError(f"Unsupported comparator: {comparator}")

    # Always include the comparator used in the result
    result["comparator"] = comparator.value
    if comparator == Comparator.ALLCLOSE:
        result["rtol"] = rtol
        result["atol"] = atol

    return result
