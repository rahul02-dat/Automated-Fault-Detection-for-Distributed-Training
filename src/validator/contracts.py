from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StateScope(str, Enum):
    """Semantic scope of a state item across a distributed system."""
    GLOBAL = "global"          # One logical value must agree across all ranks
    REPLICATED = "replicated"  # Every rank owns an equivalent copy
    SHARDED = "sharded"        # Rank-local values differ, but represent one global state
    PER_RANK = "per_rank"      # Values are intentionally rank-specific
    LOCAL = "local"            # State is process-local, not expected to match other ranks


class Comparator(str, Enum):
    """How to determine equality for a state item."""
    EXACT = "exact"            # Strict equality (==)
    ALLCLOSE = "allclose"      # Floating point tolerance equality
    HASH = "hash"              # Hash digest equality
    NORM = "norm"              # Norm-based equality (often for diagnostics)
    CUSTOM = "custom"          # Structured custom equality


@dataclass(frozen=True)
class StateContract:
    """A formal declaration of a state item's semantics for validation."""
    name: str
    scope: StateScope
    comparator: Comparator
    required: bool = True
    rtol: Optional[float] = None
    atol: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        # Validation for tolerances
        if self.comparator == Comparator.ALLCLOSE:
            if self.rtol is None or self.atol is None:
                raise ValueError(
                    f"Contract '{self.name}' uses ALLCLOSE comparator but is missing "
                    "rtol or atol."
                )
