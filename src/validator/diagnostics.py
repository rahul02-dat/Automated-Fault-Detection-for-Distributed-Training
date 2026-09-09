"""
First-mismatch localization and detailed comparison diagnostics.

Provides rich mismatch reports with: state name, nested key path, rank,
type, shape, dtype, expected/actual digests, max/mean absolute difference,
max relative difference, and first mismatch index.
"""
import torch
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .hashing import compute_digest


@dataclass
class DetailedMismatchReport:
    """A rich report of a single state mismatch."""
    state_name: str
    key_path: str = ""
    rank: int = 0
    value_type: str = ""
    shape: Optional[str] = None
    dtype: Optional[str] = None
    expected_digest: str = ""
    actual_digest: str = ""
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    first_mismatch_index: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}


def generate_detailed_report(
    state_name: str,
    expected: Any,
    actual: Any,
    rank: int = 0,
    key_path: str = "",
) -> DetailedMismatchReport:
    """
    Generate a detailed mismatch report for two state values.

    Recursively descends into dicts/lists/tuples to find the first mismatch
    and report it with full context.
    """
    full_path = f"{state_name}.{key_path}" if key_path else state_name

    # Type mismatch
    if type(expected) != type(actual):
        return DetailedMismatchReport(
            state_name=state_name,
            key_path=key_path,
            rank=rank,
            value_type=f"{type(expected).__name__} vs {type(actual).__name__}",
            reason=f"Type mismatch at {full_path}: {type(expected).__name__} != {type(actual).__name__}",
        )

    # Tensor comparison
    if isinstance(expected, torch.Tensor):
        report = DetailedMismatchReport(
            state_name=state_name,
            key_path=key_path,
            rank=rank,
            value_type="Tensor",
            shape=str(tuple(expected.shape)),
            dtype=str(expected.dtype),
        )

        if expected.shape != actual.shape:
            report.reason = f"Shape mismatch at {full_path}: {expected.shape} != {actual.shape}"
            return report

        if expected.dtype != actual.dtype:
            report.reason = f"Dtype mismatch at {full_path}: {expected.dtype} != {actual.dtype}"
            return report

        try:
            report.expected_digest = compute_digest(expected)[:16]
            report.actual_digest = compute_digest(actual)[:16]
        except Exception:
            pass

        if not torch.equal(expected, actual):
            diff = torch.abs(expected.float() - actual.float())
            report.max_abs_diff = torch.max(diff).item()
            report.mean_abs_diff = torch.mean(diff).item()

            # Safe relative difference
            denom = torch.abs(expected.float()) + 1e-8
            rel_diff = diff / denom
            report.max_rel_diff = torch.max(rel_diff).item()

            # First mismatch index
            ne_mask = (expected != actual)
            if ne_mask.any():
                idx = ne_mask.nonzero(as_tuple=False)[0]
                report.first_mismatch_index = str(tuple(idx.tolist()))

            report.reason = (
                f"Tensor mismatch at {full_path}: "
                f"max_abs_diff={report.max_abs_diff:.6e}, "
                f"max_rel_diff={report.max_rel_diff:.6e}"
            )

        return report

    # Dict comparison
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            missing = actual.keys() - expected.keys()
            extra = expected.keys() - actual.keys()
            return DetailedMismatchReport(
                state_name=state_name,
                key_path=key_path,
                rank=rank,
                value_type="dict",
                reason=f"Dict key mismatch at {full_path}. Missing: {missing}, Extra: {extra}",
            )
        for k in sorted(expected.keys()):
            child_path = f"{key_path}.{k}" if key_path else str(k)
            child_report = generate_detailed_report(state_name, expected[k], actual[k], rank, child_path)
            if child_report.reason:
                return child_report

        return DetailedMismatchReport(state_name=state_name, key_path=key_path, rank=rank, value_type="dict")

    # List/tuple comparison
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return DetailedMismatchReport(
                state_name=state_name,
                key_path=key_path,
                rank=rank,
                value_type=type(expected).__name__,
                reason=f"Length mismatch at {full_path}: {len(expected)} != {len(actual)}",
            )
        for i, (va, vb) in enumerate(zip(expected, actual)):
            child_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
            child_report = generate_detailed_report(state_name, va, vb, rank, child_path)
            if child_report.reason:
                return child_report

        return DetailedMismatchReport(
            state_name=state_name, key_path=key_path, rank=rank, value_type=type(expected).__name__
        )

    # Scalar comparison
    if expected != actual:
        return DetailedMismatchReport(
            state_name=state_name,
            key_path=key_path,
            rank=rank,
            value_type=type(expected).__name__,
            reason=f"Value mismatch at {full_path}: {expected} != {actual}",
        )

    return DetailedMismatchReport(state_name=state_name, key_path=key_path, rank=rank, value_type=type(expected).__name__)
