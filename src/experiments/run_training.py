import argparse
import os
import yaml
import json
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads.tiny_transformer import TinyTransformerWorkload
from src.workloads.tiny_transformer import TinyTransformerWorkload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()
    world_size = get_world_size()

    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'run')}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    workload_cfg = config.get("workload", {})
    if workload_cfg.get("name") == "tiny_transformer":
        workload = TinyTransformerWorkload(
            seed=seed,
            variant=workload_cfg.get("config", {}).get("variant", "fixed")
        )
    else:
        raise ValueError(f"Unknown workload: {workload_cfg.get('name')}")

    model = workload.build_model()
    ema = workload.build_ema(model)
    optimizer = workload.build_optimizer(model)
    scheduler = workload.build_scheduler(optimizer)

    ckpt_backend = TorchCheckpointBackend(
        experiment_id=config.get("experiment_id", "run"),
        run_id="run_0",
        workload_name=workload_cfg.get("name"),
        seed=seed
    )

    total_steps = config.get("training", {}).get("total_steps", 100)
    ckpt_steps = set(config.get("training", {}).get("checkpoint_steps", []))
    
    ctx = {
        "model": model,
        "ema": ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": world_size
    }

    for step in range(total_steps):
        ctx["global_step"] = step
        workload.train_step(model=model, optimizer=optimizer, ema=ema, step=step)
        
        # In typical setups, step is step+1 at the end of the loop iteration.
        # Checkpointing at specific boundaries.
        if (step + 1) in ckpt_steps:
            ctx["global_step"] = step + 1
            ckpt_path = os.path.join(out_dir, f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, ckpt_path)

    # Save final reference
    ctx["global_step"] = total_steps
    final_path = os.path.join(out_dir, "checkpoint_final")
    ckpt_backend.save(ctx, final_path)

    cleanup()

if __name__ == "__main__":
    main()
