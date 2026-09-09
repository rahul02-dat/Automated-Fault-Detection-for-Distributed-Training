from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StateScope(str, Enum):
    """
    Semantic scope of a state item across a distributed system.

    Operational semantics:

    | Scope      | Meaning                        | Validation Rule             | Expected Failure                |
    |------------|--------------------------------|-----------------------------|---------------------------------|
    | GLOBAL     | One logical value              | Consensus / equality        | Disagreement across ranks       |
    | REPLICATED | Equivalent copy on every rank  | Digest / equality           | Divergence between rank copies  |
    | PER_RANK   | Intentionally rank-specific    | Rank-specific reference     | Wrong rank state vs reference   |
    | SHARDED    | Partitioned logical state      | Coverage + consistency      | Missing/overlap/corrupt shard   |
    | LOCAL      | Intentionally process-local    | No cross-rank equality      | Should not fail (ranks differ)  |
    """
    GLOBAL = "global"          # One logical value must agree across all ranks
    REPLICATED = "replicated"  # Every rank owns an equivalent copy
    SHARDED = "sharded"        # Rank-local values differ, but represent one global state
    PER_RANK = "per_rank"      # Values are intentionally rank-specific
    LOCAL = "local"            # State is process-local, not expected to match other ranks


class Comparator(str, Enum):
    """
    How to determine equality for a state item.

    Supported comparators:
      EXACT    — Strict equality (==)
      ALLCLOSE — Floating point tolerance equality (rtol, atol required)
      HASH     — Hash digest equality via SHA-256

    Unsupported (will raise UnsupportedComparatorError at configuration time):
      NORM     — Not implemented. Reserved for future use.
      CUSTOM   — Not implemented. Reserved for future use.

    Using an unsupported comparator will fail early with a clear error message.
    """
    EXACT = "exact"            # Strict equality (==)
    ALLCLOSE = "allclose"      # Floating point tolerance equality
    HASH = "hash"              # Hash digest equality
    NORM = "norm"              # UNSUPPORTED — reserved for future extensibility
    CUSTOM = "custom"          # UNSUPPORTED — reserved for future extensibility

    @property
    def is_supported(self) -> bool:
        return self in (Comparator.EXACT, Comparator.ALLCLOSE, Comparator.HASH)


# Comparators that are not yet implemented
_UNSUPPORTED_COMPARATORS = {Comparator.NORM, Comparator.CUSTOM}


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

        # Fail early for unsupported comparators
        if self.comparator in _UNSUPPORTED_COMPARATORS:
            from .errors import UnsupportedComparatorError
            raise UnsupportedComparatorError(
                f"Contract '{self.name}' uses comparator '{self.comparator.value}' "
                f"which is not implemented. Supported comparators: "
                f"{[c.value for c in Comparator if c.is_supported]}. "
                f"Either switch to a supported comparator or remove this contract."
            )
