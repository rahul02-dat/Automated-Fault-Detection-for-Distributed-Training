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

    def apply(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        from src.runtime import distributed as dist
        rank = dist.get_rank()
        
        if rank > 0:
            if isinstance(context, dict) and "ema" in context:
                ema = context["ema"]
                if hasattr(ema, "step"):
                    if isinstance(ema.step, int):
                        ema.step = 0
                    else:
                        ema.step.zero_()
                
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "fault": self.name,
            "target": "ema.step",
            "rank_scope": "rank>0",
            "deterministic": True
        }
