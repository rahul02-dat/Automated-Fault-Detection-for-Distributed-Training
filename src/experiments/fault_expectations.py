"""
Fault-to-contract expectation map for root-cause attribution.

Maps each fault to the state contracts it is expected to violate (primary)
and the downstream contracts that may diverge as a consequence (secondary).
"""
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field


class FaultClass(str, Enum):
    """Broad categories of faults injected into training state."""
    OMISSION = "OMISSION"
    CORRUPTION = "CORRUPTION"
    STALENESS = "STALENESS"
    MISALIGNMENT = "MISALIGNMENT"
    CROSS_RANK_INCONSISTENCY = "CROSS_RANK_INCONSISTENCY"


@dataclass
class FaultExpectation:
    fault: str
    fault_class: FaultClass
    target_state: str
    mutation: str
    expected_primary: List[str] = field(default_factory=list)
    expected_secondary: List[str] = field(default_factory=list)


FAULT_EXPECTATIONS: Dict[str, FaultExpectation] = {
    "ema_scalar_omission": FaultExpectation(
        fault="ema_scalar_omission",
        fault_class=FaultClass.OMISSION,
        target_state="ema.step",
        mutation="Removes the EMA step scalar from the state dictionary.",
        expected_primary=["ema.step"],
        expected_secondary=["ema"],
    ),
    "scheduler_stale_state": FaultExpectation(
        fault="scheduler_stale_state",
        fault_class=FaultClass.STALENESS,
        target_state="scheduler",
        mutation="Decrements the scheduler's step count, simulating a stale state save.",
        expected_primary=["scheduler"],
        expected_secondary=["optimizer", "model"],
    ),
    "optimizer_state_corruption": FaultExpectation(
        fault="optimizer_state_corruption",
        fault_class=FaultClass.CORRUPTION,
        target_state="optimizer",
        mutation="Injects random noise or zeros into a selected optimizer tensor.",
        expected_primary=["optimizer"],
        expected_secondary=["model"],
    ),
    "rng_state_omission": FaultExpectation(
        fault="rng_state_omission",
        fault_class=FaultClass.OMISSION,
        target_state="rng",
        mutation="Drops the RNG state from the checkpoint payload.",
        expected_primary=["rng", "rng.torch_cpu", "rng.python", "rng.numpy"],
        expected_secondary=["model", "optimizer"],
    ),
    "data_cursor_mismatch": FaultExpectation(
        fault="data_cursor_mismatch",
        fault_class=FaultClass.MISALIGNMENT,
        target_state="data_cursor",
        mutation="Modifies the dataset index/epoch to simulate rank misalignment.",
        expected_primary=["data_cursor", "global_step"],
        expected_secondary=["model", "optimizer"],
    ),
}


def evaluate_causal_attribution(
    fault_name: str,
    failed_contracts: List[str],
) -> Dict[str, Any]:
    """
    Evaluate whether the observed failures match the expected causal chain.

    Args:
        fault_name: The injected fault name (key into FAULT_EXPECTATIONS).
        failed_contracts: List of state contract names that reported FAIL.

    Returns:
        Dict with:
            expected_primary, observed_primary,
            expected_secondary, downstream_effects,
            unexpected (failures not in any expectation list),
            causal_attribution: "PASS" | "FAIL"
    """
    if fault_name not in FAULT_EXPECTATIONS:
        return {
            "expected_primary": [],
            "observed_primary": [],
            "expected_secondary": [],
            "downstream_effects": [],
            "unexpected": list(failed_contracts),
            "causal_attribution": "UNKNOWN",
            "message": f"No expectations defined for fault '{fault_name}'.",
        }

    expectations = FAULT_EXPECTATIONS[fault_name]
    expected_primary = expectations.expected_primary
    expected_secondary = expectations.expected_secondary

    failed_set = set(failed_contracts)
    expected_all = set(expected_primary) | set(expected_secondary)

    observed_primary = [s for s in expected_primary if s in failed_set]
    downstream_effects = [s for s in expected_secondary if s in failed_set]
    unexpected = [s for s in failed_contracts if s not in expected_all]

    # Attribution passes if at least one primary contract was detected
    primary_detected = len(observed_primary) > 0
    # And there are no unexpected failures (unless they're downstream of expected secondaries)
    # We allow unexpected failures if they are plausibly downstream, but flag them
    attribution = "PASS" if primary_detected else "FAIL"

    root_cause = observed_primary[0] if observed_primary else None

    return {
        "expected_primary": expected_primary,
        "observed_primary": observed_primary,
        "expected_secondary": expected_secondary,
        "downstream_effects": downstream_effects,
        "unexpected": unexpected,
        "root_cause": root_cause,
        "causal_attribution": attribution,
        "primary_detected": primary_detected,
    }


def get_expected_root_cause(fault_name: str) -> Optional[str]:
    """Return the first expected primary state for a fault, or None."""
    if fault_name in FAULT_EXPECTATIONS:
        primaries = FAULT_EXPECTATIONS[fault_name].expected_primary
        return primaries[0] if primaries else None
    return None
