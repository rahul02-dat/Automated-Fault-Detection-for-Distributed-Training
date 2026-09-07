from typing import Any, Dict
from src.faults.base import FaultInjector
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

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "ema.step",
            "rank_scope": "rank>0",
            "deterministic": True
        }
