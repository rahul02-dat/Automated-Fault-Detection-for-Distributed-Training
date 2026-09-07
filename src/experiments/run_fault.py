import argparse
import os
import yaml
import torch
import shutil

from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.checkpoint.manifest import CheckpointManifest
from src.faults import (
    EMAScalarOmissionFault,
    SchedulerStaleStateFault,
    RNGStateOmissionFault,
    DataCursorMismatchFault,
    OptimizerStateCorruptionFault
)

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

    fault_name = config.get("fault")
    if not fault_name or fault_name not in FAULTS:
        raise ValueError(f"Unknown fault: {fault_name}")

    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'run')}")
    resume_step = config.get("training", {}).get("resume_steps", [0])[0]
    
    src_ckpt_dir = os.path.join(out_dir, f"checkpoint_{resume_step}")
    
    # We will create a mutated checkpoint directory
    mutated_ckpt_dir = os.path.join(out_dir, f"checkpoint_{resume_step}_mutated_{fault_name}")
    if rank == 0:
        os.makedirs(mutated_ckpt_dir, exist_ok=True)
        
    # Barrier to ensure directory is created
    import torch.distributed as dist
    dist.barrier()

    # 1. Load the payload directly from disk
    state_file = os.path.join(src_ckpt_dir, f"state_rank{rank}.pt")
    state = torch.load(state_file, weights_only=False)

    # 2. Inject the fault offline
    fault_injector = FAULTS[fault_name]()
    state = fault_injector.apply(state)
    
    # 3. Save the mutated payload
    out_state_file = os.path.join(mutated_ckpt_dir, f"state_rank{rank}.pt")
    torch.save(state, out_state_file)
    
    # 4. Copy the manifest over, potentially updating metadata if we wanted to
    src_manifest = os.path.join(src_ckpt_dir, f"manifest_rank{rank}.json")
    out_manifest = os.path.join(mutated_ckpt_dir, f"manifest_rank{rank}.json")
    shutil.copyfile(src_manifest, out_manifest)
    
    if rank == 0:
        print(f"[{fault_name}] Successfully mutated checkpoint offline: {mutated_ckpt_dir}")

    cleanup()

if __name__ == "__main__":
    main()
