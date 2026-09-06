import torch
from typing import Any, Dict
from .base import FaultInjector
from ..runtime import distributed as dist

class SchedulerStaleStateFault(FaultInjector):
    """
    Simulates a bug where the scheduler state is restored incorrectly,
    resulting in a stale state (e.g. step counter not advancing).
    We can simulate this by zeroing or subtracting the scheduler's last_epoch
    or step count on ranks > 0.
    """
    @property
    def name(self) -> str:
        return "scheduler_stale_state"

    def apply(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        # Mutate the scheduler in place if we can, or modify state dict.
        # In our experiments, we instantiate objects and apply faults.
        rank = dist.get_rank()
        if rank > 0:
            if "scheduler" in context:
                sched = context["scheduler"]
                if hasattr(sched, "last_epoch"):
                    sched.last_epoch = 0
                if hasattr(sched, "step_num"):
                    sched.step_num = 0
                if hasattr(sched, "_step_count"):
                    sched._step_count = 1
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "target": "scheduler",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Sets scheduler step to 0 on ranks > 0."
        }
