"""Run an uninterrupted reference training run and two resumed runs (single
resume, double resume) over 4 simulated ranks, for either the buggy or the
fixed EMA/checkpoint implementation, and report per-rank post-restore EMA
step counters plus final EMA teacher weights for comparison.

Usage:
    python harness.py --variant buggy --out /tmp/out_buggy
    python harness.py --variant fixed --out /tmp/out_fixed
"""
import argparse
import importlib
import json
import os
import shutil

import torch
import torch.nn as nn

from data import get_batch
from model import TinyTransformer

NUM_RANKS = 4
TOTAL_STEPS = 300
RESUME_1 = 80
RESUME_2 = 200
LR = 0.05
SEED = 0


def load_variant(name: str):
    ema_mod = importlib.import_module(f"ema_{name}")
    ckpt_mod = importlib.import_module(f"checkpoint_{name}")
    return ema_mod.EMAWrapper, ckpt_mod.save_checkpoint, ckpt_mod.load_checkpoint


def train_one_step(model, ema, optimizer, step):
    x, y = get_batch(step)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(model)


def new_model_and_ema(ema_cls):
    model = TinyTransformer(seed=SEED)
    ema = ema_cls(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    return model, ema, optimizer


def run_reference(ema_cls):
    model, ema, optimizer = new_model_and_ema(ema_cls)
    for step in range(TOTAL_STEPS):
        train_one_step(model, ema, optimizer, step)
    return model, ema


def run_resumed(ema_cls, save_checkpoint, load_checkpoint, ckpt_root, resume_points):
    replicas = [new_model_and_ema(ema_cls) for _ in range(NUM_RANKS)]
    post_restore_steps = {p: [None] * NUM_RANKS for p in resume_points}

    segment_bounds = [0] + list(resume_points) + [TOTAL_STEPS]
    for seg_idx in range(len(segment_bounds) - 1):
        start, end = segment_bounds[seg_idx], segment_bounds[seg_idx + 1]

        if start != 0:
            # Simulate a real process restart: brand-new model/EMA objects
            # (step counter back at its default) rather than reusing the
            # live in-memory replicas from before the "crash". Without this,
            # save+load on the same live objects never exercises the bug --
            # the in-memory step is already correct and restoring on top of
            # it trivially looks fine.
            replicas = [new_model_and_ema(ema_cls) for _ in range(NUM_RANKS)]
            ckpt_path = os.path.join(ckpt_root, f"step{start}")
            for rank in range(NUM_RANKS):
                model, ema, optimizer = replicas[rank]
                restored_step = load_checkpoint(rank, model, ema, ckpt_path)
                assert restored_step == start
                post_restore_steps[start][rank] = ema.step

        for step in range(start, end):
            for rank in range(NUM_RANKS):
                model, ema, optimizer = replicas[rank]
                train_one_step(model, ema, optimizer, step)

        if end in resume_points:
            ckpt_path = os.path.join(ckpt_root, f"step{end}")
            for rank in range(NUM_RANKS):
                model, ema, optimizer = replicas[rank]
                save_checkpoint(rank, end, model, ema, ckpt_path)

    return replicas, post_restore_steps


def run_all(variant: str, out_dir: str):
    ema_cls, save_checkpoint, load_checkpoint = load_variant(variant)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.join(out_dir, "_ckpts")
    if os.path.exists(ckpt_root):
        shutil.rmtree(ckpt_root)

    ref_model, ref_ema = run_reference(ema_cls)
    torch.save(ref_ema.state_dict(), os.path.join(out_dir, "reference_ema.pt"))
    torch.save(ref_model.state_dict(), os.path.join(out_dir, "reference_student.pt"))

    replicas_first, steps_first = run_resumed(
        ema_cls, save_checkpoint, load_checkpoint, ckpt_root, [RESUME_1]
    )
    for rank in range(NUM_RANKS):
        model, ema, _ = replicas_first[rank]
        torch.save(ema.state_dict(), os.path.join(out_dir, f"resume_first_ema_rank{rank}.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, f"resume_first_student_rank{rank}.pt"))

    replicas_both, steps_both = run_resumed(
        ema_cls, save_checkpoint, load_checkpoint, ckpt_root, [RESUME_1, RESUME_2]
    )
    for rank in range(NUM_RANKS):
        model, ema, _ = replicas_both[rank]
        torch.save(ema.state_dict(), os.path.join(out_dir, f"resume_both_ema_rank{rank}.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, f"resume_both_student_rank{rank}.pt"))

    result = {
        "variant": variant,
        "post_restore_steps": {
            "resume_first": steps_first[RESUME_1],
            "resume_both_first": steps_both[RESUME_1],
            "resume_both_second": steps_both[RESUME_2],
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    shutil.rmtree(ckpt_root, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["buggy", "fixed"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = run_all(args.variant, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
