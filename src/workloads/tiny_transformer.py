import torch
import torch.nn as nn
from typing import Any
from src.workloads.common import Workload
from src.validator.contracts import StateContract, StateScope, Comparator
from src.validator.registry import StateRegistry
from src.model import TinyTransformer
from src.data import get_batch
from src.ema_fixed import EMAWrapper as EMAFixed
from src.ema_buggy import EMAWrapper as EMABuggy

class TinyTransformerWorkload(Workload):
    def __init__(self, seed: int = 0, variant: str = "fixed", lr: float = 0.05):
        self.seed = seed
        self.variant = variant
        self.lr = lr
        
    def build_model(self) -> Any:
        return TinyTransformer(seed=self.seed)

    def build_optimizer(self, model: Any) -> Any:
        return torch.optim.SGD(model.parameters(), lr=self.lr)

    def build_scheduler(self, optimizer: Any) -> Any:
        # A mock scheduler for fault injection test coverage if needed
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    def build_data(self) -> Any:
        # data generation is purely stateless in the original code via get_batch(step)
        # return a mock dataset or simply handle it in train_step
        return None

    def build_ema(self, model: Any) -> Any:
        if self.variant == "buggy":
            return EMABuggy(model)
        return EMAFixed(model)

    def train_step(self, model: Any, optimizer: Any, ema: Any, step: int, **kwargs) -> Any:
        x, y = get_batch(step)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        return loss

    def evaluate(self, *args, **kwargs) -> Any:
        pass

    def register_state_contracts(self, registry: StateRegistry, context: Any) -> None:
        registry.register(
            StateContract(
                name="model",
                scope=StateScope.REPLICATED,
                comparator=Comparator.ALLCLOSE,
                rtol=1e-4,
                atol=1e-6
            ),
            lambda ctx: ctx.model.state_dict()
        )
        
        registry.register(
            StateContract(
                name="ema",
                scope=StateScope.REPLICATED,
                comparator=Comparator.ALLCLOSE,
                rtol=1e-4,
                atol=1e-6
            ),
            lambda ctx: ctx.ema.state_dict() if hasattr(ctx.ema, 'state_dict') else None
        )
        
        registry.register(
            StateContract(
                name="ema.step",
                scope=StateScope.GLOBAL,
                comparator=Comparator.EXACT
            ),
            lambda ctx: getattr(ctx.ema, 'step', 0) if isinstance(getattr(ctx.ema, 'step', 0), int) else getattr(ctx.ema, 'step', torch.tensor(0)).item()
        )
        
        registry.register(
            StateContract(
                name="optimizer",
                scope=StateScope.REPLICATED,
                comparator=Comparator.ALLCLOSE,
                rtol=1e-4,
                atol=1e-6
            ),
            lambda ctx: ctx.optimizer.state_dict()
        )
        
        registry.register(
            StateContract(
                name="scheduler",
                scope=StateScope.REPLICATED,
                comparator=Comparator.EXACT
            ),
            lambda ctx: ctx.scheduler.state_dict()
        )
        
        registry.register(
            StateContract(
                name="global_step",
                scope=StateScope.GLOBAL,
                comparator=Comparator.EXACT
            ),
            lambda ctx: ctx.global_step
        )
        
        registry.register(
            StateContract(
                name="rng.torch_cpu",
                scope=StateScope.REPLICATED,
                comparator=Comparator.EXACT
            ),
            lambda ctx: torch.get_rng_state()
        )
