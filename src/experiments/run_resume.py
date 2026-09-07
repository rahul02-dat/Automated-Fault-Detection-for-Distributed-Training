import argparse
import os
import yaml
import json
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads.tiny_transformer import TinyTransformerWorkload
from src.validator.registry import StateRegistry
from src.validator.validator import validate_cross_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-path", required=False, help="Path to checkpoint to resume from")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if validation fails")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()
    world_size = get_world_size()

    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'run')}")
    run_name = config.get("run_name", "resume")
    resume_dir = os.path.join(out_dir, run_name)
    
    if rank == 0:
        os.makedirs(resume_dir, exist_ok=True)
        
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
        run_id=run_name,
        workload_name=workload_cfg.get("name"),
        seed=seed
    )
    
    ctx = {
        "model": model,
        "ema": ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": world_size
    }

    # Decide which checkpoint to load
    ckpt_path = args.ckpt_path or config.get("resume_ckpt_path")
    if not ckpt_path:
        resume_step = config.get("training", {}).get("resume_steps", [0])[0]
        ckpt_path = os.path.join(out_dir, f"checkpoint_{resume_step}")
    
    # 1. Load the checkpoint
    manifest = ckpt_backend.load(ctx, ckpt_path)
    resume_step = ctx.get("global_step", 0)
    
    # 2. Validate state immediately upon restore
    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)
    
    # Expected state from baseline can be passed if we want strictly EXACT checks against baseline
    validation_results = validate_cross_rank(registry, ctx)
    if rank == 0:
        val_path = os.path.join(resume_dir, f"validation_restore_{resume_step}.json")
        with open(val_path, "w") as f:
            json.dump(validation_results, f, indent=2)

    if args.strict:
        has_failure = any(res.get("status") == "FAIL" for res in validation_results)
        if has_failure:
            cleanup()
            raise RuntimeError("Validation failed in strict mode.")

    # 3. Train the remaining steps
    total_steps = config.get("training", {}).get("total_steps", 100)
    ckpt_steps = set(config.get("training", {}).get("checkpoint_steps", []))

    for step in range(resume_step, total_steps):
        workload.train_step(model=model, optimizer=optimizer, ema=ema, scheduler=scheduler, step=step)
        ctx["global_step"] = step + 1
        
        # Checkpointing
        if (step + 1) in ckpt_steps:
            new_ckpt_path = os.path.join(resume_dir, f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, new_ckpt_path)

    # Validate at the end
    final_results = validate_cross_rank(registry, ctx)
    if rank == 0:
        final_val_path = os.path.join(resume_dir, f"validation_final_{total_steps}.json")
        with open(final_val_path, "w") as f:
            json.dump(final_results, f, indent=2)
            
        # Also compute standard evaluation and save it
        eval_metric = workload.evaluate(model, ema=ema)
        eval_path = os.path.join(resume_dir, f"eval_final_{total_steps}.json")
        with open(eval_path, "w") as f:
            json.dump({"loss": eval_metric}, f, indent=2)
            
    # Save final reference
    ctx["global_step"] = total_steps
    final_path = os.path.join(resume_dir, "checkpoint_final")
    ckpt_backend.save(ctx, final_path)

    cleanup()

if __name__ == "__main__":
    main()
