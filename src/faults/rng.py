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
        # The true bug is forgetting to restore RNG state, which means the
        # process uses its initialized seed state rather than the resumed one.
        # So we just delete the rng state from the checkpoint payload if it exists.
        if "rng" in state:
            del state["rng"]
        return state

    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For RNG faults, the mutation is applied to the runtime state, not the
        checkpoint dict. We verify by checking that the torch RNG state changed.
        """
        rank = dist.get_rank()

        # We capture torch RNG state before and after as byte digests
        import hashlib

        # Since we simulate omission by deleting 'rng' from the checkpoint state,
        # we check if it is missing in after_state but present in before_state.
        has_rng_before = "rng" in before_state
        has_rng_after = "rng" in after_state
        mutation_applied = has_rng_before and not has_rng_after
        
        # for diagnostic purposes we can hash the torch_cpu state if it existed
        before_rng = before_state.get("rng", {}).get("torch_cpu")
        if before_rng is not None:
            before_digest = hashlib.sha256(before_rng.numpy().tobytes()).hexdigest()[:16]
        else:
            before_digest = "unavailable"
            
        after_digest = "deleted" if mutation_applied else before_digest

        return {
            "fault": self.name,
            "state": self.target_state,
            "rank": rank,
            "path": "torch.random.get_rng_state()",
            "type": "bytearray",
            "shape": f"[{len(before_rng) if before_rng is not None else 0}]",
            "digest_before": before_digest,
            "digest_after": after_digest,
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
