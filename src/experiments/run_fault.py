"""
Fault injection experiment runner.

Loads a healthy checkpoint, applies exactly one fault, verifies the mutation
was applied, and saves both the mutated checkpoint and a fault_mutation.json
report.
"""
import argparse
import copy
import os
import json
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

def _deep_copy_state(state):
    """Create a deep copy of a state dict, handling tensors correctly."""
    copied = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            copied[k] = v.clone()
        elif isinstance(v, dict):
            copied[k] = _deep_copy_state(v)
        elif isinstance(v, (list, tuple)):
            copied[k] = type(v)(_deep_copy_state({"_": x})["_"] if isinstance(x, dict) else (x.clone() if isinstance(x, torch.Tensor) else x) for x in v)
        else:
            try:
                copied[k] = copy.deepcopy(v)
            except Exception:
                copied[k] = v
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()

    fault_name = config.get("fault", "none")
    if fault_name != "none" and fault_name not in FAULTS:
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

    # 2. Capture pre-mutation state for verification
    before_state = _deep_copy_state(state)

    # For RNG faults, also capture the RNG state before
    rng_before = None
    if fault_name == "rng_state_omission":
        rng_before = torch.get_rng_state()

    # 3. Inject the fault offline
    mutation_report = None
    if fault_name != "none":
        fault_injector = FAULTS[fault_name]()
        state = fault_injector.apply(state)

        # Capture post-mutation state for verification
        after_state = _deep_copy_state(state)

        # For RNG faults, add RNG state snapshots to the states
        if fault_name == "rng_state_omission":
            before_state["_rng_torch_cpu_before"] = rng_before
            after_state["_rng_torch_cpu_after"] = torch.get_rng_state()

        # Verify the mutation was actually applied
        mutation_report = fault_injector.verify_mutation(before_state, after_state)

    # 4. Save the mutated payload
    out_state_file = os.path.join(mutated_ckpt_dir, f"state_rank{rank}.pt")
    torch.save(state, out_state_file)

    # 5. Copy the manifest over
    src_manifest = os.path.join(src_ckpt_dir, f"manifest_rank{rank}.json")
    out_manifest = os.path.join(mutated_ckpt_dir, f"manifest_rank{rank}.json")
    shutil.copyfile(src_manifest, out_manifest)

    # 6. Save the mutation verification report
    if mutation_report is not None:
        # Make report JSON-serializable
        serializable_report = {}
        for k, v in mutation_report.items():
            if isinstance(v, torch.Tensor):
                serializable_report[k] = v.tolist()
            else:
                serializable_report[k] = v

        report_path = os.path.join(mutated_ckpt_dir, f"fault_mutation_rank{rank}.json")
        with open(report_path, "w") as f:
            json.dump(serializable_report, f, indent=2, default=str)

        if rank == 0:
            if mutation_report.get("mutation_applied", False):
                print(f"[{fault_name}] Mutation VERIFIED: {mutation_report}")
            else:
                print(f"[{fault_name}] WARNING: Mutation NOT verified! Report: {mutation_report}")

    if rank == 0:
        print(f"[{fault_name}] Successfully mutated checkpoint offline: {mutated_ckpt_dir}")

    cleanup()

if __name__ == "__main__":
    main()
