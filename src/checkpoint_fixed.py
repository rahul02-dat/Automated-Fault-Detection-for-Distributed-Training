import os
import pickle

import torch


def save_checkpoint(rank: int, global_step: int, model, ema, path: str):
    """Each rank saves its own shard. All EMA state -- including the step
    counter -- now lives inside ema.state_dict() and is captured here
    through the same tensor-serialization path used for every parameter."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f"student_rank{rank}.pt"))
    torch.save(ema.state_dict(), os.path.join(path, f"ema_rank{rank}.pt"))
    with open(os.path.join(path, f"global_step_rank{rank}.pkl"), "wb") as f:
        pickle.dump(global_step, f)


def load_checkpoint(rank: int, model, ema, path: str) -> int:
    """Restore a rank's shard. ema.load_state_dict restores the step counter
    identically and unconditionally on every rank -- there is no separate,
    rank-gated code path for it anymore."""
    model.load_state_dict(torch.load(os.path.join(path, f"student_rank{rank}.pt")))
    ema.load_state_dict(torch.load(os.path.join(path, f"ema_rank{rank}.pt")))

    with open(os.path.join(path, f"global_step_rank{rank}.pkl"), "rb") as f:
        global_step = pickle.load(f)

    return global_step
