import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from src.data import get_batch
from src.model import TinyTransformer

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

def run_segment(ema_cls, save_checkpoint, load_checkpoint, start, end, ckpt_root, save_points, rank):
    model, ema, optimizer = new_model_and_ema(ema_cls)
    restored_step = None
    post_restore_ema_step = None
    if start != 0:
        ckpt_path = os.path.join(ckpt_root, f"step{start}")
        restored_step = load_checkpoint(rank, model, ema, ckpt_path)
        assert restored_step == start, f"rank {rank} expected step {start}, got {restored_step}"
        post_restore_ema_step = ema.step
        print(f"[DEBUG] rank {rank} loaded step {start}, ema.step is now {post_restore_ema_step}")
        # We need to sync the RNG state if there was any stochasticity, but here it's deterministic.
    
    for step in range(start, end):
        train_one_step(model, ema, optimizer, step)
        
    if end in save_points:
        ckpt_path = os.path.join(ckpt_root, f"step{end}")
        print(f"[DEBUG] rank {rank} saving step {end}, ema.step is currently {ema.step}")
        save_checkpoint(rank, end, model, ema, ckpt_path)

    return model, ema, restored_step, post_restore_ema_step

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["buggy", "fixed"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["reference", "resume_first", "resume_both"], required=True)
    args = parser.parse_args()

    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ema_cls, save_checkpoint, load_checkpoint = load_variant(args.variant)
    out_dir = args.out
    ckpt_root = os.path.join(out_dir, "_ckpts")

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ckpt_root, exist_ok=True)
    dist.barrier()

    if args.mode == "reference":
        model, ema, _, _ = run_segment(ema_cls, save_checkpoint, load_checkpoint, 0, TOTAL_STEPS, ckpt_root, [], rank)
        if rank == 0:
            torch.save(ema.state_dict(), os.path.join(out_dir, "reference_ema.pt"))
            torch.save(model.state_dict(), os.path.join(out_dir, "reference_student.pt"))
            
    elif args.mode == "resume_first":
        # 0 to RESUME_1
        run_segment(ema_cls, save_checkpoint, load_checkpoint, 0, RESUME_1, ckpt_root, [RESUME_1], rank)
        dist.barrier()
        # RESUME_1 to TOTAL_STEPS
        model, ema, restored_step, post_restore_ema_step = run_segment(ema_cls, save_checkpoint, load_checkpoint, RESUME_1, TOTAL_STEPS, ckpt_root, [], rank)
        torch.save(ema.state_dict(), os.path.join(out_dir, f"resume_first_ema_rank{rank}.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, f"resume_first_student_rank{rank}.pt"))
        # Save the restored step so test can read it
        with open(os.path.join(out_dir, f"resume_first_step_rank{rank}.json"), "w") as f:
            json.dump({"restored_step": restored_step, "ema_step": post_restore_ema_step}, f)
            
    elif args.mode == "resume_both":
        # 0 to RESUME_1
        run_segment(ema_cls, save_checkpoint, load_checkpoint, 0, RESUME_1, ckpt_root, [RESUME_1], rank)
        dist.barrier()
        # RESUME_1 to RESUME_2
        _, ema_first, restored_step_first, post_restore_ema_step_first = run_segment(ema_cls, save_checkpoint, load_checkpoint, RESUME_1, RESUME_2, ckpt_root, [RESUME_2], rank)
        dist.barrier()
        # RESUME_2 to TOTAL_STEPS
        model, ema_second, restored_step_second, post_restore_ema_step_second = run_segment(ema_cls, save_checkpoint, load_checkpoint, RESUME_2, TOTAL_STEPS, ckpt_root, [], rank)
        
        torch.save(ema_second.state_dict(), os.path.join(out_dir, f"resume_both_ema_rank{rank}.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, f"resume_both_student_rank{rank}.pt"))
        
        with open(os.path.join(out_dir, f"resume_both_steps_rank{rank}.json"), "w") as f:
            json.dump({"restored_step_first": restored_step_first, "ema_step_first": post_restore_ema_step_first,
                       "restored_step_second": restored_step_second, "ema_step_second": post_restore_ema_step_second}, f)

    dist.destroy_process_group()

if __name__ == "__main__":
    run_worker()
