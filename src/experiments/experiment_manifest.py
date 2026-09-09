"""
Experiment manifest — sufficient metadata to reconstruct the experiment configuration.

Each run writes a manifest.json alongside its raw results.
"""
import json
import os
import datetime
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..runtime.environment import collect_environment, get_git_sha
from .result_schema import compute_config_hash


@dataclass
class ExperimentManifest:
    """
    Complete metadata for an experiment run.

    Written to: results/raw/<experiment_id>/<run_id>/manifest.json
    """
    experiment_id: str
    run_id: str = ""

    # --- Configuration ---
    config_hash: str = ""
    git_sha: str = ""
    timestamp: str = ""
    workload: str = ""
    fault: str = "none"
    world_size: int = 1
    backend: str = "gloo"
    device: str = "cpu"
    seed: int = 42
    checkpoint_step: int = 0
    total_steps: int = 0

    # --- Software versions ---
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""

    # --- Hardware ---
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    # --- Full environment snapshot ---
    environment: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        if not self.git_sha:
            self.git_sha = get_git_sha()

    @classmethod
    def from_config(cls, config: Dict[str, Any], run_id: str = "") -> "ExperimentManifest":
        """Build a manifest from a YAML experiment config dict."""
        import torch

        env = collect_environment()
        training = config.get("training", {})
        dist_cfg = config.get("distributed", {})
        workload_cfg = config.get("workload", {})

        cuda_version = ""
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or ""

        return cls(
            experiment_id=config.get("experiment_id", "unknown"),
            run_id=run_id,
            config_hash=compute_config_hash(config),
            git_sha=env.get("git_sha", "unknown"),
            workload=workload_cfg.get("name", "unknown"),
            fault=config.get("fault", "none"),
            world_size=dist_cfg.get("world_size", 1),
            backend=dist_cfg.get("backend", "gloo"),
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=training.get("seed", 42),
            checkpoint_step=training.get("checkpoint_steps", [0])[0] if training.get("checkpoint_steps") else 0,
            total_steps=training.get("total_steps", 0),
            python_version=sys.version,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            hardware_info={
                "gpu_model": env.get("gpu_model", "CPU"),
                "platform": env.get("platform", ""),
            },
            environment=env,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentManifest":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_to_run_dir(self, base_dir: str = "results/raw") -> str:
        run_dir = os.path.join(base_dir, self.experiment_id, self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "manifest.json")
        return self.save(path)
