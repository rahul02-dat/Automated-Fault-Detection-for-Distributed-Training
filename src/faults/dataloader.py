from typing import Any, Dict
from .base import FaultInjector
from ..runtime import distributed as dist


class DataCursorMismatchFault(FaultInjector):
    """
    Simulates a bug where the logical data position (or global step used to index data)
    is restored incorrectly.
    We simulate this by shifting the global_step on ranks > 0.
    """

    @property
    def name(self) -> str:
        return "data_cursor_mismatch"

    @property
    def target_state(self) -> str:
        return "global_step"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()
        if rank > 0:
            if "global_step" in state:
                state["global_step"] += 1
        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()

        before_val = before_state.get("global_step")
        after_val = after_state.get("global_step")

        mutation_applied = (rank > 0 and before_val != after_val) or (rank == 0)

        return {
            "fault": self.name,
            "state": self.target_state,
            "rank": rank,
            "before": before_val,
            "after": after_val,
            "mutation_applied": mutation_applied,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "global_step",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Increments the global_step on ranks > 0 after restore.",
        }
