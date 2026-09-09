from typing import Any, Dict
from src.faults.base import FaultInjector
from src.validator.hashing import compute_digest
import copy


class EMAScalarOmissionFault(FaultInjector):
    """
    Simulates a bug where the EMA step scalar is restored only on rank 0,
    or omitted entirely, causing cross-rank divergence when the EMA update
    rate differs.
    """

    @property
    def name(self) -> str:
        return "ema_scalar_omission"

    @property
    def target_state(self) -> str:
        return "ema.step"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from src.runtime import distributed as dist
        rank = dist.get_rank()

        # Only inject fault on ranks > 0
        if rank > 0:
            if "ema" in state:
                ema_state = state["ema"]
                if "step" in ema_state:
                    import torch
                    if isinstance(ema_state["step"], int):
                        ema_state["step"] = 0
                    elif isinstance(ema_state["step"], torch.Tensor):
                        ema_state["step"].zero_()

        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        from src.runtime import distributed as dist
        rank = dist.get_rank()

        before_val = None
        after_val = None

        if "ema" in before_state and "step" in before_state["ema"]:
            before_val = before_state["ema"]["step"]
            if hasattr(before_val, "item"):
                before_val = before_val.item()

        if "ema" in after_state and "step" in after_state["ema"]:
            after_val = after_state["ema"]["step"]
            if hasattr(after_val, "item"):
                after_val = after_val.item()

        # On rank > 0, the step should have been zeroed
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
            "target": "ema.step",
            "rank_scope": "rank>0",
            "deterministic": True,
        }
