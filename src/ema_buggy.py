import copy

import torch


class EMAWrapper:
    """Maintains an exponential-moving-average 'teacher' copy of a model.
    The decay rate ramps up over the first steps, so the step counter is
    part of the update math, not just bookkeeping.

    BUG: `step` is a plain Python int living directly on the wrapper. It is
    not part of `state_dict()`, so nothing about the checkpoint format
    guarantees it gets saved/restored consistently -- see checkpoint_buggy.py.
    """

    def __init__(self, model, max_decay: float = 0.999):
        self.module = copy.deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.max_decay = max_decay
        self.step = 0

    def decay(self):
        return min(self.max_decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, model):
        d = self.decay()
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p, alpha=1 - d)
        self.step += 1

    def state_dict(self):
        # Only tensor state is captured here -- `step` is missing.
        return {"module": self.module.state_dict()}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
