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

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()
        if rank > 0:
            if "global_step" in state:
                state["global_step"] += 1
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "target": "global_step",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Increments the global_step on ranks > 0 after restore."
        }
