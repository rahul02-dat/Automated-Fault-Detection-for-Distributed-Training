"""
Overhead benchmarking experiment runner.

Measures training wall time, checkpoint time, and validation time separately.
Supports multiple repetitions for statistical reporting.
Benchmarks both healthy-path (all digests agree) and failure-path (digest mismatch).
"""
import argparse
import os
import time
import yaml
import json
import statistics
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads import TinyTransformerWorkload, ResNetWorkload, SmallTransformerWorkload
from src.validator.registry import StateRegistry
from src.validator.validator import validate_cross_rank
from src.experiments.result_schema import ExperimentResult, ExperimentPhase, ExperimentStatus, compute_config_hash


def run_single_benchmark(config, workload, disable_validation, run_idx):
    """Run a single benchmark iteration and return timing metrics."""
    rank = get_rank()
    world_size = get_world_size()
    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    workload_cfg = config.get("workload", {})
    workload_name = workload_cfg.get("name")

    model = workload.build_model()
    ema = workload.build_ema(model) if hasattr(workload, "build_ema") else None
    optimizer = workload.build_optimizer(model)
    scheduler = workload.build_scheduler(optimizer)

    ckpt_backend = TorchCheckpointBackend(
        experiment_id=config.get("experiment_id", "run"),
        run_id=f"bench_{run_idx}",
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
        "run_idx": run_idx,
        "training_time": 0.0,
        "checkpoint_time": 0.0,
        "validation_time": 0.0,
        "total_time": 0.0,
        "step_times": [],
        "total_steps": total_steps,
        "world_size": world_size,
        "validation_enabled": not disable_validation,
    }

    total_start = time.time()

    for step in range(total_steps):
        step_start = time.time()

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

        step_time = time.time() - step_start
        metrics["step_times"].append(step_time)

        # Checkpointing
        if (step + 1) in ckpt_steps:
            t0 = time.time()
            out_dir = config.get("output_dir", f"results/benchmarks/{config.get('experiment_id', 'run')}")
            ckpt_path = os.path.join(out_dir, f"bench_{run_idx}", f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, ckpt_path)
            ckpt_time = time.time() - t0
            metrics["checkpoint_time"] += ckpt_time

    metrics["training_time"] = time.time() - total_start - metrics["checkpoint_time"]

    # Final validation
    if not disable_validation:
        val_start = time.time()
        registry = StateRegistry()
        workload.register_state_contracts(registry, ctx)
        final_results = validate_cross_rank(registry, ctx)
        metrics["validation_time"] = time.time() - val_start
    else:
        final_results = []

    metrics["total_time"] = time.time() - total_start

    return metrics, final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--disable-validation", action="store_true",
                       help="Run without cross-rank validation to measure baseline overhead.")
    parser.add_argument("--repetitions", type=int, default=5,
                       help="Number of repetitions for statistical reporting (default: 5).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()
    world_size = get_world_size()

    out_dir = config.get("output_dir", f"results/benchmarks/{config.get('experiment_id', 'run')}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    workload_cfg = config.get("workload", {})
    workload_name = workload_cfg.get("name")

    if workload_name == "tiny_transformer":
        workload = TinyTransformerWorkload(
            seed=config.get("training", {}).get("seed", 0),
            variant=workload_cfg.get("config", {}).get("variant", "fixed")
        )
    elif workload_name == "resnet":
        workload = ResNetWorkload(config=workload_cfg.get("config", {}))
    elif workload_name == "small_transformer":
        workload = SmallTransformerWorkload(config=workload_cfg.get("config", {}))
    else:
        raise ValueError(f"Unknown workload: {workload_name}")

    all_metrics = []
    for rep in range(args.repetitions):
        metrics, final_results = run_single_benchmark(
            config, workload, args.disable_validation, rep
        )
        all_metrics.append(metrics)

        if rank == 0 and final_results:
            val_path = os.path.join(out_dir, f"validation_rep{rep}.json")
            with open(val_path, "w") as f:
                json.dump(final_results, f, indent=2)

    if rank == 0:
        # Compute aggregate statistics
        mode = "disabled" if args.disable_validation else "enabled"

        training_times = [m["training_time"] for m in all_metrics]
        checkpoint_times = [m["checkpoint_time"] for m in all_metrics]
        validation_times = [m["validation_time"] for m in all_metrics]
        total_times = [m["total_time"] for m in all_metrics]

        def compute_stats(values):
            if not values:
                return {}
            s = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
            if len(values) >= 20:
                sorted_vals = sorted(values)
                idx = int(0.95 * len(sorted_vals))
                s["p95"] = sorted_vals[idx]
            return s

        aggregate = {
            "mode": mode,
            "repetitions": args.repetitions,
            "workload": workload_name,
            "world_size": world_size,
            "total_steps": config.get("training", {}).get("total_steps", 100),
            "training_time": compute_stats(training_times),
            "checkpoint_time": compute_stats(checkpoint_times),
            "validation_time": compute_stats(validation_times),
            "total_time": compute_stats(total_times),
            "per_run_metrics": all_metrics,
        }

        agg_path = os.path.join(out_dir, f"benchmark_metrics_{mode}.json")
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        print(f"Benchmark ({mode}): {args.repetitions} reps, "
              f"train={aggregate['training_time'].get('mean', 0):.4f}±{aggregate['training_time'].get('std', 0):.4f}s, "
              f"ckpt={aggregate['checkpoint_time'].get('mean', 0):.4f}s, "
              f"val={aggregate['validation_time'].get('mean', 0):.4f}s")

    cleanup()

if __name__ == "__main__":
    main()
