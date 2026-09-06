import copy

import torch


class EMAWrapper:
    """Maintains an exponential-moving-average 'teacher' copy of a model.
    The decay rate ramps up over the first steps, so the step counter is
    part of the update math, not just bookkeeping.

    FIX: `step` is backed by a 0-dim `torch.long` buffer instead of a plain
    Python int, so it lives inside `state_dict()` and is captured/restored
    through exactly the same path -- with exactly the same synchronization
    guarantees -- as every other piece of tensor state.
    """

    def __init__(self, model, max_decay: float = 0.999):
        self.module = copy.deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.max_decay = max_decay
        self._step_buf = torch.zeros((), dtype=torch.long)

    @property
    def step(self) -> int:
        return int(self._step_buf.item())

    @step.setter
    def step(self, value: int):
        self._step_buf.fill_(int(value))

    def decay(self):
        return min(self.max_decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, model):
        d = self.decay()
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p, alpha=1 - d)
        self._step_buf += 1

    def state_dict(self):
        return {"module": self.module.state_dict(), "step": self._step_buf.clone()}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self._step_buf.copy_(sd["step"])
