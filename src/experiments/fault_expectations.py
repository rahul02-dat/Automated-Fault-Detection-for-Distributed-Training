"""
Fault-to-contract expectation map for root-cause attribution.

Maps each fault to the state contracts it is expected to violate (primary)
and the downstream contracts that may diverge as a consequence (secondary).
"""
from typing import Any, Dict, List, Optional


FAULT_EXPECTATIONS: Dict[str, Dict[str, List[str]]] = {
    "ema_scalar_omission": {
        "primary": ["ema.step"],
        "secondary": ["ema"],
    },
    "scheduler_stale_state": {
        "primary": ["scheduler"],
        "secondary": ["optimizer", "model"],
    },
    "optimizer_state_corruption": {
        "primary": ["optimizer"],
        "secondary": ["model"],
    },
    "rng_state_omission": {
        "primary": ["rng", "rng.torch_cpu", "rng.python", "rng.numpy"],
        "secondary": ["model", "optimizer"],
    },
    "data_cursor_mismatch": {
        "primary": ["data_cursor", "global_step"],
        "secondary": ["model", "optimizer"],
    },
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
            expected_secondary, observed_secondary,
            unexpected (failures not in any expectation list),
            causal_attribution: "PASS" | "FAIL"
    """
    if fault_name not in FAULT_EXPECTATIONS:
        return {
            "expected_primary": [],
            "observed_primary": [],
            "expected_secondary": [],
            "observed_secondary": [],
            "unexpected": list(failed_contracts),
            "causal_attribution": "UNKNOWN",
            "message": f"No expectations defined for fault '{fault_name}'.",
        }

    expectations = FAULT_EXPECTATIONS[fault_name]
    expected_primary = expectations.get("primary", [])
    expected_secondary = expectations.get("secondary", [])

    failed_set = set(failed_contracts)
    expected_all = set(expected_primary) | set(expected_secondary)

    observed_primary = [s for s in expected_primary if s in failed_set]
    observed_secondary = [s for s in expected_secondary if s in failed_set]
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
        "observed_secondary": observed_secondary,
        "unexpected": unexpected,
        "root_cause_state": root_cause,
        "causal_attribution": attribution,
        "primary_detected": primary_detected,
    }


def get_expected_root_cause(fault_name: str) -> Optional[str]:
    """Return the first expected primary state for a fault, or None."""
    if fault_name in FAULT_EXPECTATIONS:
        primaries = FAULT_EXPECTATIONS[fault_name].get("primary", [])
        return primaries[0] if primaries else None
    return None
