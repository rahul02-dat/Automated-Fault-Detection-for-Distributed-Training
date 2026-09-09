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

    @property
    def target_state(self) -> str:
        return "scheduler"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Mutate the scheduler in place if we can, or modify state dict.
        rank = dist.get_rank()
        if rank > 0:
            if "scheduler" in state:
                sched = state["scheduler"]
                if "last_epoch" in sched:
                    sched["last_epoch"] = 0
                if "_step_count" in sched:
                    sched["_step_count"] = 1
        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()

        before_epoch = None
        after_epoch = None
        before_step_count = None
        after_step_count = None

        if "scheduler" in before_state:
            before_epoch = before_state["scheduler"].get("last_epoch")
            before_step_count = before_state["scheduler"].get("_step_count")
        if "scheduler" in after_state:
            after_epoch = after_state["scheduler"].get("last_epoch")
            after_step_count = after_state["scheduler"].get("_step_count")

        mutation_applied = (rank > 0 and (before_epoch != after_epoch or before_step_count != after_step_count)) or (rank == 0)

        return {
            "fault": self.name,
            "state": self.target_state,
            "rank": rank,
            "before": {"last_epoch": before_epoch, "_step_count": before_step_count},
            "after": {"last_epoch": after_epoch, "_step_count": after_step_count},
            "mutation_applied": mutation_applied,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "scheduler",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Sets scheduler step to 0 on ranks > 0.",
        }
