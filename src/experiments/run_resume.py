import argparse
import os
import yaml
import json
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads import TinyTransformerWorkload, ResNetWorkload, SmallTransformerWorkload
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
    workload_name = workload_cfg.get("name")
    
    if workload_name == "tiny_transformer":
        workload = TinyTransformerWorkload(
            seed=seed,
            variant=workload_cfg.get("config", {}).get("variant", "fixed")
        )
    elif workload_name == "resnet":
        workload = ResNetWorkload(config=workload_cfg.get("config", {}))
    elif workload_name == "small_transformer":
        workload = SmallTransformerWorkload(config=workload_cfg.get("config", {}))
    else:
        raise ValueError(f"Unknown workload: {workload_name}")

    model = workload.build_model()
    ema = workload.build_ema(model) if hasattr(workload, "build_ema") else None
    optimizer = workload.build_optimizer(model)
    scheduler = workload.build_scheduler(optimizer)

    ckpt_backend = TorchCheckpointBackend(
        experiment_id=config.get("experiment_id", "run"),
        run_id=run_name,
        workload_name=workload_name,
        seed=seed
    )
    
    ctx = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": world_size
    }
    if ema is not None:
        ctx["ema"] = ema

    # Setup dataloader if available
    data_res = workload.build_data()
    if data_res is not None:
        if isinstance(data_res, tuple):
            dataloader, sampler = data_res
        else:
            dataloader = data_res
        data_iter = iter(dataloader)
    else:
        data_iter = None

    # Decide which checkpoint to load
    ckpt_path = args.ckpt_path or config.get("resume_ckpt_path")
    if not ckpt_path:
        resume_step = config.get("training", {}).get("resume_steps", [0])[0]
        ckpt_path = os.path.join(out_dir, f"checkpoint_{resume_step}")
    
    # 1. Load the checkpoint
    manifest = ckpt_backend.load(ctx, ckpt_path)
    resume_step = ctx.get("global_step", 0)
    
    # Fast-forward dataloader if needed (simplified for testing)
    if data_iter is not None and resume_step > 0:
        for _ in range(resume_step):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                next(data_iter)
    
    # 2. Validate state immediately upon restore
    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)
    
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
        kwargs = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "step": step}
        if ema is not None:
            kwargs["ema"] = ema
            
        if data_iter is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            kwargs["batch"] = batch

        workload.train_step(**kwargs)
        ctx["global_step"] = step + 1
        
        # Checkpointing
        if (step + 1) in ckpt_steps:
            new_ckpt_path = os.path.join(resume_dir, f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, new_ckpt_path)

    # Validate at the end
    final_results = validate_cross_rank(registry, ctx)
    
    # Also compute standard evaluation and save it
    eval_metric = None
    if hasattr(workload, "evaluate"):
        eval_kwargs = {"model": model}
        if ema is not None:
            eval_kwargs["ema"] = ema
        if data_res is not None:
            eval_kwargs["dataloader"] = dataloader
        eval_metric = workload.evaluate(**eval_kwargs)
        
    if rank == 0:
        final_val_path = os.path.join(resume_dir, f"validation_final_{total_steps}.json")
        with open(final_val_path, "w") as f:
            json.dump(final_results, f, indent=2)
            
        if eval_metric is not None:
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
