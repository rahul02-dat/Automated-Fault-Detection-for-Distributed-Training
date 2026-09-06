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

    def apply(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        rank = dist.get_rank()
        if rank > 0:
            if isinstance(context, dict) and "optimizer" in context:
                opt = context["optimizer"]
                for param_state in opt.state.values():
                    if "momentum_buffer" in param_state:
                        # Zero out momentum buffer
                        param_state["momentum_buffer"].zero_()
                        break
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "target": "optimizer",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Zeroes out a momentum buffer in the optimizer on ranks > 0."
        }
