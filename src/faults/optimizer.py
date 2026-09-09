import torch
from typing import Any, Dict
from .base import FaultInjector
from ..runtime import distributed as dist
from ..validator.hashing import compute_digest


class OptimizerStateCorruptionFault(FaultInjector):
    """
    Simulates a bug where optimizer state is corrupted or omitted upon restore.
    We simulate this by zeroing out the learning rate for all parameter groups
    on ranks > 0.
    """

    @property
    def name(self) -> str:
        return "optimizer_state_corruption"

    @property
    def target_state(self) -> str:
        return "optimizer"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()
        if rank > 0:
            if isinstance(state, dict) and "optimizer" in state:
                opt = state["optimizer"]
                if "param_groups" in opt:
                    for group in opt["param_groups"]:
                        group["lr"] = 0.0
        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        rank = dist.get_rank()

        before_lr = None
        after_lr = None
        before_digest = None
        after_digest = None

        if "optimizer" in before_state and "param_groups" in before_state["optimizer"]:
            before_lr = [g.get("lr") for g in before_state["optimizer"]["param_groups"]]
            try:
                before_digest = compute_digest(before_state["optimizer"])
            except Exception:
                before_digest = "error"

        if "optimizer" in after_state and "param_groups" in after_state["optimizer"]:
            after_lr = [g.get("lr") for g in after_state["optimizer"]["param_groups"]]
            try:
                after_digest = compute_digest(after_state["optimizer"])
            except Exception:
                after_digest = "error"

        mutation_applied = (rank > 0 and before_lr != after_lr) or (rank == 0)

        return {
            "fault": self.name,
            "state": self.target_state,
            "rank": rank,
            "before": {"lr": before_lr, "digest": before_digest},
            "after": {"lr": after_lr, "digest": after_digest},
            "mutation_applied": mutation_applied,
            "mutation": "lr_zeroed",
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "optimizer",
            "scope": "cross_rank",
            "deterministic": True,
            "description": "Zeroes out the learning rate in the optimizer on ranks > 0.",
        }
