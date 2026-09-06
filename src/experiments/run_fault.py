import argparse
import os
import yaml
import json
from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads.tiny_transformer import TinyTransformerWorkload
from src.faults import (
    EMAScalarOmissionFault,
    SchedulerStaleStateFault,
    RNGStateOmissionFault,
    DataCursorMismatchFault,
    OptimizerStateCorruptionFault
)
from src.validator.registry import StateRegistry
from src.validator.validator import validate_cross_rank

FAULTS = {
    "ema_scalar_omission": EMAScalarOmissionFault,
    "scheduler_stale_state": SchedulerStaleStateFault,
    "rng_state_omission": RNGStateOmissionFault,
    "data_cursor_mismatch": DataCursorMismatchFault,
    "optimizer_state_corruption": OptimizerStateCorruptionFault
}

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

    # Resume directory should match output_dir from run_training
    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'run')}")
    fault_dir = os.path.join(out_dir, "fault_run")
    if rank == 0:
        os.makedirs(fault_dir, exist_ok=True)
        
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
    
    ctx = {
        "model": model,
        "ema": ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": world_size
    }

    resume_step = config.get("training", {}).get("resume_steps", [0])[0]
    ckpt_path = os.path.join(out_dir, f"checkpoint_{resume_step}")
    
    # 1. Load the checkpoint
    manifest = ckpt_backend.load(ctx, ckpt_path)
    
    # 2. Inject the fault
    fault_name = config.get("fault")
    if fault_name and fault_name in FAULTS:
        fault_injector = FAULTS[fault_name]()
        fault_injector.apply(None, ctx)
                    
    # 3. Validate state
    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)
    
    validation_results = validate_cross_rank(registry, ctx)
    if rank == 0:
        val_path = os.path.join(fault_dir, f"validation_restore_{resume_step}.json")
        with open(val_path, "w") as f:
            json.dump(validation_results, f, indent=2)

    # 4. Train a few steps to show resume divergence
    total_steps = config.get("training", {}).get("total_steps", 100)
    for step in range(resume_step, total_steps):
        ctx["global_step"] = step
        workload.train_step(model=model, optimizer=optimizer, ema=ema, step=step)

    # Validate at the end to show divergence (e.g., if fault silently propagated)
    ctx["global_step"] = total_steps
    final_results = validate_cross_rank(registry, ctx)
    if rank == 0:
        final_val_path = os.path.join(fault_dir, f"validation_final_{total_steps}.json")
        with open(final_val_path, "w") as f:
            json.dump(final_results, f, indent=2)

    cleanup()

if __name__ == "__main__":
    main()
