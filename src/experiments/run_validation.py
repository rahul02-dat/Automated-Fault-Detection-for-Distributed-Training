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
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to validate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()
    world_size = get_world_size()

    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'run')}")
    val_dir = os.path.join(out_dir, "validation_run")
    if rank == 0:
        os.makedirs(val_dir, exist_ok=True)
        
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

    ckpt_path = os.path.join(out_dir, f"checkpoint_{args.step}")
    
    # Load the checkpoint
    ckpt_backend.load(ctx, ckpt_path)
                    
    # Validate state
    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)
    
    validation_results = validate_cross_rank(registry, ctx)
    if rank == 0:
        val_path = os.path.join(val_dir, f"validation_step_{args.step}.json")
        with open(val_path, "w") as f:
            json.dump(validation_results, f, indent=2)

    cleanup()

if __name__ == "__main__":
    main()
