import os
import pickle

import torch


def save_checkpoint(rank: int, global_step: int, model, ema, path: str):
    """Each rank saves its own shard of the checkpoint. Tensor state (student
    and EMA teacher parameters) goes through torch.save. The EMA wrapper's
    scalar step counter is a plain Python int, so it's serialized separately
    via pickle, outside the tensor state_dict."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f"student_rank{rank}.pt"))
    torch.save(ema.state_dict(), os.path.join(path, f"ema_rank{rank}.pt"))
    with open(os.path.join(path, f"ema_step_rank{rank}.pkl"), "wb") as f:
        pickle.dump(ema.step, f)
    with open(os.path.join(path, f"global_step_rank{rank}.pkl"), "wb") as f:
        pickle.dump(global_step, f)


def load_checkpoint(rank: int, model, ema, path: str) -> int:
    """Restore a rank's shard. Tensor state is restored identically on every
    rank via ema.load_state_dict(). The scalar EMA step counter, however, is
    only reliably applied on rank 0 -- this mirrors a real failure mode in
    sharded-checkpoint restore paths (e.g. FSDP's SHARDED_STATE_DICT), where
    tensor synchronization is guaranteed across ranks but arbitrary Python
    object state is not. Ranks 1-3 silently keep whatever step value they
    already had in memory."""
    model.load_state_dict(torch.load(os.path.join(path, f"student_rank{rank}.pt")))
    ema.load_state_dict(torch.load(os.path.join(path, f"ema_rank{rank}.pt")))

    with open(os.path.join(path, f"global_step_rank{rank}.pkl"), "rb") as f:
        global_step = pickle.load(f)

    if rank == 0:
        with open(os.path.join(path, f"ema_step_rank{rank}.pkl"), "rb") as f:
            ema.step = pickle.load(f)

    return global_step
