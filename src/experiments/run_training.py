"""
Baseline training experiment runner.

Trains a workload from scratch, saves checkpoints at specified steps,
and emits ExperimentResult + ExperimentManifest JSON records.
"""
import argparse
import os
import time
import yaml
import json
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads import TinyTransformerWorkload, ResNetWorkload, SmallTransformerWorkload
from src.experiments.result_schema import ExperimentResult, ExperimentPhase, ExperimentStatus, compute_config_hash
from src.experiments.experiment_manifest import ExperimentManifest


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
        run_id="run_0",
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

    # Timing
    train_start = time.time()
    checkpoint_time = 0.0

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

        # global_step invariant: number of completed optimizer updates
        ctx["global_step"] = step + 1

        # Checkpointing at specific boundaries.
        if (step + 1) in ckpt_steps:
            t0 = time.time()
            ckpt_path = os.path.join(out_dir, f"checkpoint_{step + 1}")
            ckpt_backend.save(ctx, ckpt_path)
            checkpoint_time += (time.time() - t0)

    # Save final reference
    ctx["global_step"] = total_steps
    t0 = time.time()
    final_path = os.path.join(out_dir, "checkpoint_final")
    ckpt_backend.save(ctx, final_path)
    checkpoint_time += (time.time() - t0)

    training_duration = time.time() - train_start

    # Final validation and eval
    from src.validator.registry import StateRegistry
    from src.validator.validator import validate_cross_rank

    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)

    val_start = time.time()
    final_results = validate_cross_rank(registry, ctx)
    validation_duration_ms = (time.time() - val_start) * 1000

    # Optional eval
    eval_metric = None
    if hasattr(workload, "evaluate"):
        eval_kwargs = {"model": model}
        if ema is not None:
            eval_kwargs["ema"] = ema
        if data_res is not None:
            eval_kwargs["dataloader"] = dataloader
        eval_metric = workload.evaluate(**eval_kwargs)

    if rank == 0:
        # Save validation results
        final_val_path = os.path.join(out_dir, f"validation_final_{total_steps}.json")
        with open(final_val_path, "w") as f:
            json.dump(final_results, f, indent=2)

        if eval_metric is not None:
            eval_path = os.path.join(out_dir, "eval_final.json")
            with open(eval_path, "w") as f:
                json.dump({"loss": eval_metric}, f, indent=2)

        # Emit ExperimentResult
        from src.runtime.environment import collect_environment
        result = ExperimentResult(
            experiment_id=config.get("experiment_id", "run"),
            run_id="run_0",
            workload=workload_name,
            fault="none",
            phase=ExperimentPhase.BASELINE.value,
            world_size=world_size,
            backend=config.get("distributed", {}).get("backend", "gloo"),
            device="cuda" if next(model.parameters()).is_cuda else "cpu",
            seed=seed,
            checkpoint_step=list(ckpt_steps)[0] if ckpt_steps else 0,
            total_steps=total_steps,
            validation_enabled=True,
            status=ExperimentStatus.PASS.value if all(
                r.get("status") != "FAIL" for r in final_results
            ) else ExperimentStatus.FAIL.value,
            detected=any(r.get("status") == "FAIL" for r in final_results),
            detected_states=[r["state_name"] for r in final_results if r.get("status") == "FAIL"],
            validation_duration_ms=validation_duration_ms,
            training_duration_s=training_duration,
            checkpoint_duration_s=checkpoint_time,
            final_metric=eval_metric,
            config_hash=compute_config_hash(config),
            environment=collect_environment(),
            validation_results=final_results,
        )
        result.save(os.path.join(out_dir, "raw_results.json"))

        # Emit ExperimentManifest
        manifest = ExperimentManifest.from_config(config, run_id="run_0")
        manifest.save(os.path.join(out_dir, "manifest.json"))

    cleanup()

if __name__ == "__main__":
    main()
