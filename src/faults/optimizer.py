import torch
from typing import Any, Dict
from .base import FaultInjector
from ..runtime import distributed as dist

class OptimizerStateCorruptionFault(FaultInjector):
    """
    Simulates a bug where optimizer state is corrupted or omitted upon restore.
    We simulate this by zeroing out the momentum buffer for one of the parameters on ranks > 0.
    """
    @property
    def name(self) -> str:
        return "optimizer_state_corruption"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()
        if rank > 0:
            if isinstance(state, dict) and "optimizer" in state:
                opt = state["optimizer"]
                if "param_groups" in opt:
                    for group in opt["param_groups"]:
                        group["lr"] = 0.0
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "target": "optimizer",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Zeroes out a momentum buffer in the optimizer on ranks > 0."
        }
