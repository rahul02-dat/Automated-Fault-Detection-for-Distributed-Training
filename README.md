# EMA Checkpoint-Resume Bug: Reproduction and Fix

A minimal, runnable demonstration of a real class of distributed-training bug:
a sharded checkpoint format that synchronizes tensor state correctly across
ranks but silently fails to synchronize a piece of scalar Python state.

## The scenario

A training pipeline maintains an exponential-moving-average (EMA) "teacher"
copy of a student model — a standard pattern in self-distillation / semi-
supervised setups. The EMA decay rate ramps up over the first steps:

```
decay = min(max_decay, (1 + step) / (10 + step))
```

`step` here is the EMA wrapper's own update counter, so it's part of the
actual math, not just bookkeeping.

Training works fine end-to-end. But if training is stopped and resumed from a
checkpoint partway through — a routine operation — the final EMA teacher
weights come out different from an uninterrupted run, and different *between
ranks*, even though the student model resumes identically everywhere.

## The bug

`src/ema_buggy.py` stores `step` as a plain Python `int` attribute — not part
of `state_dict()`. `src/checkpoint_buggy.py` saves it out-of-band via pickle,
and on restore only re-applies it on rank 0:

```python
if rank == 0:
    with open(...) as f:
        ema.step = pickle.load(f)
```

Every other rank silently keeps whatever `step` it already had (its default,
after a real process restart). Every rank then computes a different decay
rate for the same physical step, so the EMA teacher shards drift apart —
invisible in the student model, invisible in any single-rank loss curve, and
only visible once you compare the reconstructed teacher weights across ranks.

## The fix

`src/ema_fixed.py` backs `step` with a 0-dim `torch.long` buffer, so it lives
inside `EMAWrapper.state_dict()` / `load_state_dict()` and is captured and
restored through exactly the same path — with exactly the same guarantees —
as the EMA teacher's actual parameters. `src/checkpoint_fixed.py` no longer
needs (or has) any rank-conditional branch: every rank just calls
`ema.load_state_dict(...)` unconditionally.

## Layout

```
src/
  model.py              tiny transformer student model, deterministic init
  data.py                fixed-seed synthetic batches (pure function of step)
  ema_buggy.py            EMA wrapper -- step as a plain int (the bug)
  ema_fixed.py             EMA wrapper -- step as a registered buffer (the fix)
  checkpoint_buggy.py       per-rank save/load -- rank-0-only step restore (the bug)
  checkpoint_fixed.py        per-rank save/load -- unconditional step restore (the fix)
  harness.py                  runs an uninterrupted reference + a single-resume
                               + a double-resume scenario, 4 simulated ranks,
                               for either variant
tests/
  test_ema_checkpoint_bug.py  proves the bug reproduces (buggy variant) and
                               is resolved (fixed variant)
```

`harness.py` simulates 4 ranks in a single process for simplicity (no real
`torch.distributed`/FSDP setup required) — at each checkpoint boundary it
discards the in-memory model/EMA objects and creates fresh ones before
restoring, to genuinely simulate a process restart rather than restoring on
top of already-correct in-memory state.

## Running it

```bash
pip install -r requirements.txt

# See the bug reproduce:
python src/harness.py --variant buggy --out /tmp/out_buggy
# post_restore_steps.resume_first will look like [80, 0, 0, 0]

# See the fix resolve it:
python src/harness.py --variant fixed --out /tmp/out_fixed
# post_restore_steps.resume_first will look like [80, 80, 80, 80]

# Full test suite (bug reproduction + fix verification, weight-level checks):
pytest tests/ -v
```

All 8 tests pass on the fixed variant; the "buggy_*" tests document and pin
down the failure mode using the buggy variant.

## Verification gates (in `tests/test_ema_checkpoint_bug.py`)

- **Reproduction gate** — unpatched code shows an exact cross-rank integer
  mismatch in post-restore EMA step (no tolerance; it's a discrete counter).
- **Cross-rank counter-sync gate** — patched code shows exact equality across
  all 4 ranks, at both restore points of the double-resume run.
- **Weight-equivalence gate** — patched code's final EMA teacher weights
  match the uninterrupted reference run within `rtol=1e-4, atol=1e-6` (a band
  that only needs to absorb ordinary floating-point rounding, since the
  pipeline is fully deterministic — no dropout, fixed seeds, no real
  multi-process nondeterminism).
- **Student non-regression gate** — student weights stay bit-for-bit
  identical to the reference in both variants, confirming the fix doesn't
  touch the (already-correct) student restore path.
- **Double-resume gate** — sync and weight-equivalence both re-checked after
  *two* independent restores, which would catch a one-shot patch (e.g. a
  single post-hoc `broadcast()` right after the first observed restore) that
  happens to pass a single-resume test but doesn't generalize.
