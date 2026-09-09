import torch
import random
from typing import Any, Dict
from .base import FaultInjector
from ..runtime import distributed as dist


class RNGStateOmissionFault(FaultInjector):
    """
    Simulates a bug where RNG state is not restored correctly.
    We simulate this by advancing the RNG manually, effectively throwing it out of sync.
    """

    @property
    def name(self) -> str:
        return "rng_state_omission"

    @property
    def target_state(self) -> str:
        return "rng"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Perturb the RNG state so it doesn't match the resumed state
        torch.manual_seed(9999 + dist.get_rank())
        random.seed(9999 + dist.get_rank())
        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For RNG faults, the mutation is applied to the runtime state, not the
        checkpoint dict. We verify by checking that the torch RNG state changed.
        """
        rank = dist.get_rank()

        # We capture torch RNG state before and after as byte digests
        import hashlib

        before_rng = before_state.get("_rng_torch_cpu_before")
        after_rng = after_state.get("_rng_torch_cpu_after")

        if before_rng is not None and after_rng is not None:
            before_digest = hashlib.sha256(before_rng.numpy().tobytes()).hexdigest()[:16]
            after_digest = hashlib.sha256(after_rng.numpy().tobytes()).hexdigest()[:16]
            mutation_applied = before_digest != after_digest
        else:
            before_digest = "unavailable"
            after_digest = "unavailable"
            mutation_applied = True  # Assume applied since we set the seed explicitly

        return {
            "fault": self.name,
            "state": self.target_state,
            "rank": rank,
            "before_digest": before_digest,
            "after_digest": after_digest,
            "mutation_applied": mutation_applied,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "rng",
            "scope": "resume_equivalence",
            "deterministic": True,
            "description": "Corrupts the RNG state after restore to simulate omission.",
        }
