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

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Perturb the RNG state so it doesn't match the resumed state
        torch.manual_seed(9999 + dist.get_rank())
        random.seed(9999 + dist.get_rank())
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "target": "rng",
            "scope": "resume_equivalence",
            "deterministic": True,
            "description": "Corrupts the RNG state after restore to simulate omission."
        }
