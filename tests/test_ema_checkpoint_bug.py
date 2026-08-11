"""Proves two things:

1. The buggy pipeline (ema_buggy.py + checkpoint_buggy.py) actually
   reproduces the described failure: post-restore EMA step counters diverge
   across ranks, and that divergence compounds into a measurable EMA teacher
   weight discrepancy vs. an uninterrupted reference run.
2. The fixed pipeline (ema_fixed.py + checkpoint_fixed.py) resolves it:
   exact cross-rank counter agreement and EMA teacher weights matching the
   reference within a tight floating-point tolerance -- while the student
   model (never buggy) stays bit-for-bit identical in both variants.
"""
import shutil
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from harness import RESUME_1, RESUME_2, run_all  # noqa: E402

TMP_ROOT = Path("/tmp/ema_checkpoint_bug_test")


@pytest.fixture(scope="session")
def buggy_result():
    out = TMP_ROOT / "buggy"
    if out.exists():
        shutil.rmtree(out)
    return run_all("buggy", str(out)), out


@pytest.fixture(scope="session")
def fixed_result():
    out = TMP_ROOT / "fixed"
    if out.exists():
        shutil.rmtree(out)
    return run_all("fixed", str(out)), out


# ---------------------------------------------------------------------------
# Bug reproduction (documents the failure the fix must resolve)
# ---------------------------------------------------------------------------


def test_buggy_cross_rank_counters_diverge(buggy_result):
    """The unpatched pipeline must show at least one rank's post-restore EMA
    step differ from the others -- this confirms the harness actually
    exercises the real mechanism, not just a downstream symptom."""
    result, _ = buggy_result
    vals = result["post_restore_steps"]["resume_first"]
    assert len(set(vals)) > 1, f"expected cross-rank divergence, got {vals}"
    assert vals[0] == RESUME_1  # rank 0 is the one rank that restores correctly


def test_buggy_ema_weights_diverge_from_reference(buggy_result):
    """The corrupted EMA teacher weights on non-zero ranks must differ from
    the uninterrupted reference run by much more than floating-point noise."""
    result, out = buggy_result
    ref = torch.load(out / "reference_ema.pt")["module"]
    diverged = False
    for rank in range(1, 4):
        got = torch.load(out / f"resume_first_ema_rank{rank}.pt")["module"]
        for k in ref:
            if not torch.allclose(ref[k], got[k], rtol=1e-3, atol=1e-4):
                diverged = True
    assert diverged, "expected corrupted ranks' EMA weights to diverge from reference"


def test_buggy_student_still_matches_reference(buggy_result):
    """The student model is unaffected by the bug -- this is what makes the
    bug invisible to ordinary loss-curve or single-rank monitoring."""
    result, out = buggy_result
    ref = torch.load(out / "reference_student.pt")
    for rank in range(4):
        got = torch.load(out / f"resume_first_student_rank{rank}.pt")
        for k in ref:
            assert torch.equal(ref[k], got[k])


# ---------------------------------------------------------------------------
# Fix verification
# ---------------------------------------------------------------------------


def test_fixed_cross_rank_counters_sync_single_resume(fixed_result):
    """Exact integer equality across all 4 ranks after a single restore."""
    result, _ = fixed_result
    vals = result["post_restore_steps"]["resume_first"]
    assert len(set(vals)) == 1, f"ranks disagree: {vals}"
    assert vals[0] == RESUME_1


def test_fixed_cross_rank_counters_sync_double_resume(fixed_result):
    """Both restores of the double-resume run must show exact agreement --
    rejects one-shot patches (e.g. a single post-hoc broadcast) that would
    only happen to fix the first resume."""
    result, _ = fixed_result
    first_vals = result["post_restore_steps"]["resume_both_first"]
    second_vals = result["post_restore_steps"]["resume_both_second"]
    assert len(set(first_vals)) == 1 and first_vals[0] == RESUME_1
    assert len(set(second_vals)) == 1 and second_vals[0] == RESUME_2


def test_fixed_ema_weights_match_reference_single_resume(fixed_result):
    result, out = fixed_result
    ref = torch.load(out / "reference_ema.pt")["module"]
    for rank in range(4):
        got = torch.load(out / f"resume_first_ema_rank{rank}.pt")["module"]
        for k in ref:
            assert torch.allclose(ref[k], got[k], rtol=1e-4, atol=1e-6), (
                f"rank {rank} EMA weight {k} diverges after single resume"
            )


def test_fixed_ema_weights_match_reference_double_resume(fixed_result):
    result, out = fixed_result
    ref = torch.load(out / "reference_ema.pt")["module"]
    for rank in range(4):
        got = torch.load(out / f"resume_both_ema_rank{rank}.pt")["module"]
        for k in ref:
            assert torch.allclose(ref[k], got[k], rtol=1e-4, atol=1e-6), (
                f"rank {rank} EMA weight {k} diverges after double resume"
            )


def test_fixed_student_non_regression(fixed_result):
    """Student weights must remain bit-for-bit identical to the reference --
    confirms the fix didn't perturb the (already-correct) student restore
    path while patching the EMA one."""
    result, out = fixed_result
    ref = torch.load(out / "reference_student.pt")
    for scenario_prefix in ["resume_first_student_rank", "resume_both_student_rank"]:
        for rank in range(4):
            got = torch.load(out / f"{scenario_prefix}{rank}.pt")
            for k in ref:
                assert torch.equal(ref[k], got[k])
