import torch

from src.model import VOCAB

BATCH_SIZE = 4
SEQLEN = 8


def get_batch(step: int, seed: int = 1234):
    """Deterministic synthetic batch: a pure function of the global step
    index and a fixed seed. Identical on an uninterrupted run and on any
    resumed run at the same step, on every rank -- so the only thing that
    can make a resumed run diverge from the reference is training state
    that fails to restore correctly."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    x = torch.randint(0, VOCAB, (BATCH_SIZE, SEQLEN), generator=g)
    y = torch.randint(0, VOCAB, (BATCH_SIZE, SEQLEN), generator=g)
    return x, y
