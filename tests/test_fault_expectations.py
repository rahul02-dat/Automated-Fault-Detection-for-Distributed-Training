import pytest
from src.experiments.fault_expectations import FAULT_EXPECTATIONS, FaultClass, evaluate_causal_attribution, get_expected_root_cause

def test_fault_taxonomy_complete():
    """Ensure all required faults are present and properly structured."""
    required_faults = {
        "ema_scalar_omission",
        "scheduler_stale_state",
        "optimizer_state_corruption",
        "rng_state_omission",
        "data_cursor_mismatch",
    }
    
    # All required faults must exist
    assert required_faults.issubset(set(FAULT_EXPECTATIONS.keys()))
    
    for f_name, expectation in FAULT_EXPECTATIONS.items():
        assert expectation.fault == f_name
        assert isinstance(expectation.fault_class, FaultClass)
        assert len(expectation.expected_primary) > 0
        assert expectation.target_state != ""
        assert expectation.mutation != ""

def test_evaluate_causal_attribution():
    """Test the attribution logic with a mock failure."""
    # If ema.step and ema fail, it's a pass for ema_scalar_omission
    res = evaluate_causal_attribution("ema_scalar_omission", ["ema.step", "ema"])
    assert res["causal_attribution"] == "PASS"
    assert res["root_cause"] == "ema.step"
    assert res["primary_detected"] is True
    assert res["unexpected"] == []
    
    # If we get an unrelated failure, it's flagged as unexpected
    res2 = evaluate_causal_attribution("ema_scalar_omission", ["ema.step", "model"])
    assert res2["causal_attribution"] == "PASS" # still pass because primary is detected
    assert "model" in res2["unexpected"]
    
    # If primary is completely missed, it's a FAIL
    res3 = evaluate_causal_attribution("ema_scalar_omission", ["ema"])
    assert res3["causal_attribution"] == "FAIL"
    assert res3["primary_detected"] is False

def test_get_expected_root_cause():
    assert get_expected_root_cause("ema_scalar_omission") == "ema.step"
    assert get_expected_root_cause("unknown_fault") is None
