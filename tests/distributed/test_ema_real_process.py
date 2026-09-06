import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = TESTS_DIR / "test_ema_real_process_worker.py"
TMP_ROOT = Path("/tmp/ema_real_process_test")

NUM_RANKS = 4
RESUME_1 = 80
RESUME_2 = 200

def run_torchrun(variant: str, out_dir: Path, mode: str):
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={NUM_RANKS}",
        "--master_addr=127.0.0.1",
        "--master_port=29501",
        str(WORKER_SCRIPT),
        "--variant", variant,
        "--out", str(out_dir),
        "--mode", mode
    ]
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"torchrun failed for {variant} {mode}")

@pytest.fixture(scope="session")
def buggy_result():
    out = TMP_ROOT / "buggy"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    
    run_torchrun("buggy", out, "reference")
    run_torchrun("buggy", out, "resume_first")
    # For buggy, we don't necessarily need resume_both to prove the bug, but we can run it
    # to keep symmetry with fixed test.
    run_torchrun("buggy", out, "resume_both")
    
    return out

@pytest.fixture(scope="session")
def fixed_result():
    out = TMP_ROOT / "fixed"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    
    run_torchrun("fixed", out, "reference")
    run_torchrun("fixed", out, "resume_first")
    run_torchrun("fixed", out, "resume_both")
    
    return out

# ---------------------------------------------------------------------------
# Bug reproduction 
# ---------------------------------------------------------------------------

def test_buggy_cross_rank_counters_diverge(buggy_result):
    out = buggy_result
    steps = []
    for rank in range(NUM_RANKS):
        with open(out / f"resume_first_step_rank{rank}.json") as f:
            data = json.load(f)
            steps.append(data["ema_step"])
            
    assert len(set(steps)) > 1, f"expected cross-rank divergence, got {steps}"
    assert steps[0] == RESUME_1  # rank 0 is correct

def test_buggy_ema_weights_diverge_from_reference(buggy_result):
    out = buggy_result
    ref = torch.load(out / "reference_ema.pt", weights_only=True)["module"]
    diverged = False
    for rank in range(1, NUM_RANKS):
        got = torch.load(out / f"resume_first_ema_rank{rank}.pt", weights_only=True)["module"]
        for k in ref:
            if not torch.allclose(ref[k], got[k], rtol=1e-3, atol=1e-4):
                diverged = True
    assert diverged, "expected corrupted ranks' EMA weights to diverge from reference"

def test_buggy_student_still_matches_reference(buggy_result):
    out = buggy_result
    ref = torch.load(out / "reference_student.pt", weights_only=True)
    for rank in range(NUM_RANKS):
        got = torch.load(out / f"resume_first_student_rank{rank}.pt", weights_only=True)
        for k in ref:
            assert torch.equal(ref[k], got[k])

# ---------------------------------------------------------------------------
# Fix verification
# ---------------------------------------------------------------------------

def test_fixed_cross_rank_counters_sync_single_resume(fixed_result):
    out = fixed_result
    steps = []
    for rank in range(NUM_RANKS):
        with open(out / f"resume_first_step_rank{rank}.json") as f:
            steps.append(json.load(f)["ema_step"])
            
    assert len(set(steps)) == 1, f"ranks disagree: {steps}"
    assert steps[0] == RESUME_1

def test_fixed_cross_rank_counters_sync_double_resume(fixed_result):
    out = fixed_result
    first_steps = []
    second_steps = []
    for rank in range(NUM_RANKS):
        with open(out / f"resume_both_steps_rank{rank}.json") as f:
            data = json.load(f)
            first_steps.append(data["ema_step_first"])
            second_steps.append(data["ema_step_second"])
            
    assert len(set(first_steps)) == 1 and first_steps[0] == RESUME_1
    assert len(set(second_steps)) == 1 and second_steps[0] == RESUME_2

def test_fixed_ema_weights_match_reference_single_resume(fixed_result):
    out = fixed_result
    ref = torch.load(out / "reference_ema.pt", weights_only=True)["module"]
    for rank in range(NUM_RANKS):
        got = torch.load(out / f"resume_first_ema_rank{rank}.pt", weights_only=True)["module"]
        for k in ref:
            assert torch.allclose(ref[k], got[k], rtol=1e-4, atol=1e-6)

def test_fixed_ema_weights_match_reference_double_resume(fixed_result):
    out = fixed_result
    ref = torch.load(out / "reference_ema.pt", weights_only=True)["module"]
    for rank in range(NUM_RANKS):
        got = torch.load(out / f"resume_both_ema_rank{rank}.pt", weights_only=True)["module"]
        for k in ref:
            assert torch.allclose(ref[k], got[k], rtol=1e-4, atol=1e-6)

def test_fixed_student_non_regression(fixed_result):
    out = fixed_result
    ref = torch.load(out / "reference_student.pt", weights_only=True)
    for scenario_prefix in ["resume_first_student_rank", "resume_both_student_rank"]:
        for rank in range(NUM_RANKS):
            got = torch.load(out / f"{scenario_prefix}{rank}.pt", weights_only=True)
            for k in ref:
                assert torch.equal(ref[k], got[k])
