"""
Resume equivalence protocol (P3).

Two-run protocol:
  REFERENCE — uninterrupted training from step 0 to N.
  RESUMED   — train to step K, checkpoint, restore, train from K to N.

Compares state digests at each step from K through N and reports
trajectory divergence with precise language.
"""
import argparse
import os
import time
import json
import yaml
import torch
from typing import Any, Dict, List, Optional

from src.runtime.distributed import init_process_group, destroy_process_group as cleanup, get_rank, get_world_size
from src.runtime.seeds import set_deterministic_seeds, capture_rng_state, restore_rng_state
from src.checkpoint.torch_checkpoint import TorchCheckpointBackend
from src.workloads import TinyTransformerWorkload, ResNetWorkload, SmallTransformerWorkload
from src.validator.registry import StateRegistry
from src.validator.hashing import compute_digest
from src.validator.comparison import canonicalize
from src.experiments.result_schema import ExperimentResult, ExperimentPhase, ExperimentStatus, compute_config_hash


def _build_workload(config):
    workload_cfg = config.get("workload", {})
    workload_name = workload_cfg.get("name")
    seed = config.get("training", {}).get("seed", 0)

    if workload_name == "tiny_transformer":
        return TinyTransformerWorkload(
            seed=seed,
            variant=workload_cfg.get("config", {}).get("variant", "fixed")
        ), workload_name
    elif workload_name == "resnet":
        return ResNetWorkload(config=workload_cfg.get("config", {})), workload_name
    elif workload_name == "small_transformer":
        return SmallTransformerWorkload(config=workload_cfg.get("config", {})), workload_name
    else:
        raise ValueError(f"Unknown workload: {workload_name}")


def _extract_digests(registry: StateRegistry, ctx: Dict[str, Any]) -> Dict[str, str]:
    """Extract state digests for all registered contracts."""
    digests = {}
    for name, contract in registry.get_all_contracts().items():
        try:
            val = registry.extract_state(name, ctx)
            canon = canonicalize(val)
            digests[name] = compute_digest(canon)
        except Exception as e:
            digests[name] = f"ERROR:{e}"
    return digests


def _run_training_phase(workload, config, start_step, end_step, ctx, registry, data_iter=None, dataloader=None):
    """Run training from start_step to end_step, capturing digests at each step."""
    trajectory = []

    for step in range(start_step, end_step):
        kwargs = {
            "model": ctx["model"],
            "optimizer": ctx["optimizer"],
            "scheduler": ctx["scheduler"],
            "step": step,
        }
        if "ema" in ctx:
            kwargs["ema"] = ctx["ema"]

        if data_iter is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            kwargs["batch"] = batch

        workload.train_step(**kwargs)
        ctx["global_step"] = step + 1

        # Capture digests after this step
        digests = _extract_digests(registry, ctx)
        trajectory.append({
            "step": step + 1,
            "digests": digests,
        })

    return trajectory, data_iter


def run_reference(config, out_dir):
    """Run uninterrupted training from step 0 to N."""
    rank = get_rank()
    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    workload, workload_name = _build_workload(config)
    total_steps = config.get("training", {}).get("total_steps", 100)
    checkpoint_step = config.get("training", {}).get("checkpoint_steps", [5])[0]

    model = workload.build_model()
    ema = workload.build_ema(model) if hasattr(workload, "build_ema") else None
    optimizer = workload.build_optimizer(model)
    scheduler = workload.build_scheduler(optimizer)

    ctx = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": get_world_size(),
    }
    if ema is not None:
        ctx["ema"] = ema

    data_res = workload.build_data()
    if data_res is not None:
        if isinstance(data_res, tuple):
            dataloader, sampler = data_res
        else:
            dataloader = data_res
            sampler = None
        data_iter = iter(dataloader)
    else:
        data_iter = None
        dataloader = None

    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)

    # Run full training, capturing digests from checkpoint_step onward
    pre_trajectory, data_iter = _run_training_phase(
        workload, config, 0, checkpoint_step, ctx, registry, data_iter, dataloader
    )
    post_trajectory, data_iter = _run_training_phase(
        workload, config, checkpoint_step, total_steps, ctx, registry, data_iter, dataloader
    )

    # Evaluate final metric
    eval_metric = None
    if hasattr(workload, "evaluate"):
        eval_kwargs = {"model": model}
        if ema is not None:
            eval_kwargs["ema"] = ema
        if data_res is not None:
            eval_kwargs["dataloader"] = dataloader
        eval_metric = workload.evaluate(**eval_kwargs)

    return post_trajectory, eval_metric


def run_resumed(config, out_dir):
    """Train to K, checkpoint, restore, then train K to N."""
    rank = get_rank()
    seed = config.get("training", {}).get("seed", 0)
    set_deterministic_seeds(seed, rank)

    workload, workload_name = _build_workload(config)
    total_steps = config.get("training", {}).get("total_steps", 100)
    checkpoint_step = config.get("training", {}).get("checkpoint_steps", [5])[0]

    # Phase 1: Train from 0 to K and checkpoint
    model = workload.build_model()
    ema = workload.build_ema(model) if hasattr(workload, "build_ema") else None
    optimizer = workload.build_optimizer(model)
    scheduler = workload.build_scheduler(optimizer)

    ctx = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "rank": rank,
        "world_size": get_world_size(),
    }
    if ema is not None:
        ctx["ema"] = ema

    data_res = workload.build_data()
    if data_res is not None:
        if isinstance(data_res, tuple):
            dataloader, sampler = data_res
        else:
            dataloader = data_res
            sampler = None
        data_iter = iter(dataloader)
    else:
        data_iter = None
        dataloader = None

    registry = StateRegistry()
    workload.register_state_contracts(registry, ctx)

    # Train to K
    for step in range(checkpoint_step):
        kwargs = {
            "model": ctx["model"],
            "optimizer": ctx["optimizer"],
            "scheduler": ctx["scheduler"],
            "step": step,
        }
        if "ema" in ctx:
            kwargs["ema"] = ctx["ema"]

        if data_iter is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            kwargs["batch"] = batch

        workload.train_step(**kwargs)
        ctx["global_step"] = step + 1

    # Save checkpoint at step K
    ckpt_backend = TorchCheckpointBackend(
        experiment_id=config.get("experiment_id", "resume_equiv"),
        run_id="resumed",
        workload_name=workload_name,
        seed=seed,
    )
    ckpt_path = os.path.join(out_dir, f"checkpoint_{checkpoint_step}")
    ckpt_backend.save(ctx, ckpt_path)

    # Phase 2: Re-init everything and restore from checkpoint
    set_deterministic_seeds(seed, rank)

    model2 = workload.build_model()
    ema2 = workload.build_ema(model2) if hasattr(workload, "build_ema") else None
    optimizer2 = workload.build_optimizer(model2)
    scheduler2 = workload.build_scheduler(optimizer2)

    ctx2 = {
        "model": model2,
        "optimizer": optimizer2,
        "scheduler": scheduler2,
        "global_step": 0,
        "rank": rank,
        "world_size": get_world_size(),
    }
    if ema2 is not None:
        ctx2["ema"] = ema2

    ckpt_backend.load(ctx2, ckpt_path)

    # Re-setup dataloader and fast-forward
    data_res2 = workload.build_data()
    if data_res2 is not None:
        if isinstance(data_res2, tuple):
            dataloader2, sampler2 = data_res2
        else:
            dataloader2 = data_res2
            sampler2 = None
        data_iter2 = iter(dataloader2)
        for _ in range(checkpoint_step):
            try:
                next(data_iter2)
            except StopIteration:
                data_iter2 = iter(dataloader2)
                next(data_iter2)
    else:
        data_iter2 = None
        dataloader2 = None

    registry2 = StateRegistry()
    workload.register_state_contracts(registry2, ctx2)

    # Phase 3: Continue training from K to N
    post_trajectory, data_iter2 = _run_training_phase(
        workload, config, checkpoint_step, total_steps, ctx2, registry2, data_iter2, dataloader2
    )

    # Evaluate final metric
    eval_metric = None
    if hasattr(workload, "evaluate"):
        eval_kwargs = {"model": model2}
        if ema2 is not None:
            eval_kwargs["ema"] = ema2
        if data_res2 is not None:
            eval_kwargs["dataloader"] = dataloader2
        eval_metric = workload.evaluate(**eval_kwargs)

    return post_trajectory, eval_metric


def main():
    parser = argparse.ArgumentParser(description="Resume equivalence protocol")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_process_group(config.get("distributed", {}).get("backend", "gloo"))
    rank = get_rank()

    out_dir = config.get("output_dir", f"results/raw/{config.get('experiment_id', 'resume_equiv')}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Run reference (uninterrupted)
    ref_trajectory, ref_metric = run_reference(config, out_dir)

    # Run resumed (checkpoint + restore)
    res_trajectory, res_metric = run_resumed(config, out_dir)

    if rank == 0:
        # Build comparison table
        comparison = []
        all_match = True

        for ref_entry, res_entry in zip(ref_trajectory, res_trajectory):
            step = ref_entry["step"]
            step_match = True
            state_matches = {}

            for state_name in ref_entry["digests"]:
                ref_digest = ref_entry["digests"].get(state_name, "MISSING")
                res_digest = res_entry["digests"].get(state_name, "MISSING")
                match = ref_digest == res_digest
                state_matches[state_name] = {
                    "reference_digest": ref_digest[:16] if not ref_digest.startswith("ERROR") else ref_digest,
                    "resumed_digest": res_digest[:16] if not res_digest.startswith("ERROR") else res_digest,
                    "match": match,
                }
                if not match:
                    step_match = False
                    all_match = False

            comparison.append({
                "step": step,
                "all_match": step_match,
                "states": state_matches,
            })

        # Metric delta
        metric_delta = None
        if ref_metric is not None and res_metric is not None:
            metric_delta = abs(ref_metric - res_metric)

        report = {
            "experiment_id": config.get("experiment_id", "resume_equiv"),
            "workload": config.get("workload", {}).get("name", "unknown"),
            "checkpoint_step": config.get("training", {}).get("checkpoint_steps", [0])[0],
            "total_steps": config.get("training", {}).get("total_steps", 0),
            "world_size": get_world_size(),
            "reference_final_loss": ref_metric,
            "resumed_final_loss": res_metric,
            "absolute_delta": metric_delta,
            "relative_delta": metric_delta / abs(ref_metric) if ref_metric and ref_metric != 0 and metric_delta else None,
            "all_steps_match": all_match,
            "trajectory_comparison": comparison,
            "conclusion": (
                "All states exactly matched under the tested deterministic configuration."
                if all_match else
                "State divergence detected between reference and resumed runs."
            ),
        }

        report_path = os.path.join(out_dir, "resume_equivalence_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nResume Equivalence Report:")
        print(f"  All steps match: {all_match}")
        print(f"  Reference loss:  {ref_metric}")
        print(f"  Resumed loss:    {res_metric}")
        print(f"  Delta:           {metric_delta}")
        print(f"  Conclusion:      {report['conclusion']}")
        print(f"  Report saved:    {report_path}")

    cleanup()


if __name__ == "__main__":
    main()
