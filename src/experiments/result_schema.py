"""
Canonical experiment result schema for research-quality reproducibility.

Every experiment emits one machine-readable JSON record conforming to this schema.
Results are stable, parseable without custom assumptions, and independent of
terminal log parsing.
"""
import hashlib
import json
import os
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(str, Enum):
    """Outcome of a single experiment phase."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class ExperimentPhase(str, Enum):
    """Which phase of the experiment pipeline produced this result."""
    BASELINE = "baseline"
    FAULT_INJECTION = "fault_injection"
    RESTORE_VALIDATION = "restore_validation"
    CONTINUATION = "continuation"
    FINAL = "final"


class ExperimentOutcome(str, Enum):
    """
    Semantic outcome used by the experiment runner.
    Distinguishes correct from incorrect behaviour.
    """
    EXPECTED_DETECTION = "EXPECTED_DETECTION"       # Fault was detected correctly
    UNEXPECTED_PASS = "UNEXPECTED_PASS"             # Fault was NOT detected (false negative)
    UNEXPECTED_FAILURE = "UNEXPECTED_FAILURE"       # Healthy run incorrectly failed (false positive)
    EXPERIMENT_ERROR = "EXPERIMENT_ERROR"           # Infrastructure / setup error


@dataclass
class ExperimentResult:
    """
    A single canonical experiment result record.

    Every experiment run emits exactly one of these as JSON.
    Raw results are never overwritten — they are written to:
        results/raw/<experiment_id>/<run_id>/raw_results.json
    """
    # --- Identity ---
    experiment_id: str
    run_id: str = ""

    # --- Configuration ---
    workload: str = ""
    fault: str = "none"
    phase: str = ExperimentPhase.FINAL.value
    world_size: int = 1
    backend: str = "gloo"
    device: str = "cpu"
    seed: int = 42
    checkpoint_step: int = 0
    total_steps: int = 0
    validation_enabled: bool = True

    # --- Results ---
    status: str = ExperimentStatus.UNKNOWN.value
    outcome: str = ""
    detected: bool = False
    detected_states: List[str] = field(default_factory=list)
    root_cause_state: Optional[str] = None

    # --- Causal attribution ---
    expected_primary: List[str] = field(default_factory=list)
    observed_primary: List[str] = field(default_factory=list)
    expected_secondary: List[str] = field(default_factory=list)
    observed_secondary: List[str] = field(default_factory=list)
    causal_attribution: Optional[str] = None  # "PASS" or "FAIL"

    # --- Timing ---
    validation_duration_ms: float = 0.0
    training_duration_s: float = 0.0
    checkpoint_duration_s: float = 0.0

    # --- Metrics ---
    final_metric: Optional[float] = None
    reference_metric: Optional[float] = None
    metric_delta: Optional[float] = None

    # --- Reproducibility ---
    git_sha: str = ""
    config_hash: str = ""
    timestamp: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)

    # --- Validation detail ---
    validation_results: List[Dict[str, Any]] = field(default_factory=list)

    # --- Free-form details ---
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        return asdict(self)

    def save(self, path: str) -> str:
        """
        Save result to a JSON file.
        Creates parent directories if needed. Returns the written path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load an ExperimentResult from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_to_run_dir(self, base_dir: str = "results/raw") -> str:
        """
        Save result to the immutable run directory layout:
            results/raw/<experiment_id>/<run_id>/raw_results.json
        """
        run_dir = os.path.join(base_dir, self.experiment_id, self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "raw_results.json")
        return self.save(path)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of the experiment configuration."""
    # Sort keys recursively for determinism
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
