import argparse
import os
import yaml
import json
import time
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads import TinyTransformerWorkload, ResNetWorkload, SmallTransformerWorkload
from src.validator.registry import StateRegistry
from src.validator.validator import validate_cross_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--disable-validation", action="store_true", help="Run without cross-rank validation to measure baseline overhead.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()
    world_size = get_world_size()

    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    out_dir = config.get("output_dir", f"results/benchmarks/{config.get('experiment_id', 'run')}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    workload_cfg = config.get("workload", {})
    workload_name = workload_cfg.get("name")
    
    if workload_name == "tiny_transformer":
        workload = TinyTransformerWorkload(seed=seed, variant=workload_cfg.get("config", {}).get("variant", "fixed"))
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
        run_id="bench_0",
        workload_name=workload_name,
        seed=seed
    )

    total_steps = config.get("training", {}).get("total_steps", 100)
    ckpt_steps = set(config.get("training", {}).get("checkpoint_steps", []))
    
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

    metrics = {
        "runtime": 0.0,
        "checkpoint_time": 0.0,
        "validation_time": 0.0,
        "total_steps": total_steps,
        "world_size": world_size,
        "validation_enabled": not args.disable_validation
    }

    start_time = time.time()

    for step in range(total_steps):
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
            t0 = time.time()
            ckpt_path = os.path.join(out_dir, f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, ckpt_path)
            metrics["checkpoint_time"] += (time.time() - t0)

    train_end_time = time.time()
    metrics["runtime"] = train_end_time - start_time

    # Final validation
    if not args.disable_validation:
        val_start_time = time.time()
        registry = StateRegistry()
        workload.register_state_contracts(registry, ctx)
        final_results = validate_cross_rank(registry, ctx)
        metrics["validation_time"] = time.time() - val_start_time
        
        if rank == 0:
            final_val_path = os.path.join(out_dir, f"validation_final_{total_steps}.json")
            with open(final_val_path, "w") as f:
                json.dump(final_results, f, indent=2)

    if rank == 0:
        metrics_path = os.path.join(out_dir, f"benchmark_metrics_{'disabled' if args.disable_validation else 'enabled'}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    cleanup()

if __name__ == "__main__":
    main()
