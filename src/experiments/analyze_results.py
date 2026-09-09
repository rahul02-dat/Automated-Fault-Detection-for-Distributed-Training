"""
Automated table generation from raw experiment JSON.

Generates all 5 required research tables:
  Table A — Fault Detection
  Table B — Resume Fidelity
  Table C — Overhead
  Table D — Scalability
  Table E — Reliability
"""
import os
import json
import glob
from typing import Dict, List, Any


def _load_json_safe(path: str) -> Any:
    """Safely load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _find_result_files(base_dir: str, pattern: str = "raw_results.json") -> List[str]:
    """Find all result files matching the pattern under base_dir."""
    return glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)


# =============================================================================
# Table A — Fault Detection
# =============================================================================

def generate_fault_detection_table(raw_dir: str) -> str:
    """
    | Workload | Fault | Expected State | Restore Detection | Final Detection | Root Cause Correct | Trials |
    """
    lines = []
    lines.append("## Table A — Fault Detection")
    lines.append("")
    lines.append("| Workload | Fault | Expected State | Restore Detection | Final Detection | Root Cause Correct | Trials |")
    lines.append("|---|---|---|---|---|---|---:|")

    if not os.path.exists(raw_dir):
        lines.append("| *(No data)* | - | - | - | - | - | - |")
        return "\n".join(lines)

    # Group results by (workload, fault)
    groups: Dict[tuple, List[Dict]] = {}
    for result_path in _find_result_files(raw_dir):
        data = _load_json_safe(result_path)
        if data is None or data.get("fault", "none") == "none":
            continue
        key = (data.get("workload", "?"), data.get("fault", "?"))
        groups.setdefault(key, []).append(data)

    for (workload, fault), results in sorted(groups.items()):
        expected_primary = results[0].get("expected_primary", [])
        expected_str = ", ".join(expected_primary) if expected_primary else "?"

        # Count restore and final detections
        restore_detected = 0
        final_detected = 0
        root_cause_correct = 0
        for r in results:
            details = r.get("details", {})
            restore_val = details.get("restore_validation", [])
            if any(v.get("status") == "FAIL" for v in restore_val if isinstance(v, dict)):
                restore_detected += 1
            if r.get("detected", False):
                final_detected += 1
            if r.get("causal_attribution") == "PASS":
                root_cause_correct += 1

        n = len(results)
        lines.append(
            f"| {workload} | {fault} | {expected_str} "
            f"| {restore_detected}/{n} | {final_detected}/{n} "
            f"| {root_cause_correct}/{n} | {n} |"
        )

    return "\n".join(lines)


# =============================================================================
# Table B — Resume Fidelity
# =============================================================================

def generate_resume_fidelity_table(raw_dir: str) -> str:
    """
    | Workload | World Size | Checkpoint Step | Reference Digest | Resumed Digest | Match |
    """
    lines = []
    lines.append("## Table B — Resume Fidelity")
    lines.append("")
    lines.append("| Workload | World Size | Checkpoint Step | Reference Loss | Resumed Loss | Delta | Match |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    if not os.path.exists(raw_dir):
        lines.append("| *(No data)* | - | - | - | - | - | - |")
        return "\n".join(lines)

    for exp_dir in sorted(os.listdir(raw_dir)):
        full_dir = os.path.join(raw_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue

        # Look for eval files
        eval_files = glob.glob(os.path.join(full_dir, "**", "eval_*.json"), recursive=True)
        result_files = glob.glob(os.path.join(full_dir, "**", "raw_results.json"), recursive=True)

        if not eval_files:
            continue

        losses = []
        for ef in sorted(eval_files):
            data = _load_json_safe(ef)
            if data and "loss" in data:
                losses.append(data["loss"])

        # Get metadata from raw results
        workload = "?"
        world_size = "?"
        ckpt_step = "?"
        for rf in result_files:
            rd = _load_json_safe(rf)
            if rd:
                workload = rd.get("workload", "?")
                world_size = rd.get("world_size", "?")
                ckpt_step = rd.get("checkpoint_step", "?")
                break

        if len(losses) >= 1:
            baseline = losses[0]
            resumed = losses[-1] if len(losses) > 1 else losses[0]
            delta = abs(baseline - resumed)
            match = "Yes" if delta < 1e-5 else "No"
            lines.append(
                f"| {workload} | {world_size} | {ckpt_step} "
                f"| {baseline:.6f} | {resumed:.6f} | {delta:.2e} | {match} |"
            )

    return "\n".join(lines)


# =============================================================================
# Table C — Overhead
# =============================================================================

def generate_overhead_table(bench_dir: str) -> str:
    """
    | Workload | World Size | Validation Off | Validation On | Overhead % | Std |
    """
    lines = []
    lines.append("## Table C — Overhead")
    lines.append("")
    lines.append("| Workload | World Size | Val Off (s) | Val On (s) | Val Time (s) | Overhead % | Std % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    if not os.path.exists(bench_dir):
        lines.append("| *(No data)* | - | - | - | - | - | - |")
        return "\n".join(lines)

    for exp_dir in sorted(os.listdir(bench_dir)):
        full_dir = os.path.join(bench_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue

        dis_path = os.path.join(full_dir, "benchmark_metrics_disabled.json")
        en_path = os.path.join(full_dir, "benchmark_metrics_enabled.json")

        if os.path.exists(dis_path) and os.path.exists(en_path):
            dis_data = _load_json_safe(dis_path)
            en_data = _load_json_safe(en_path)

            if not dis_data or not en_data:
                continue

            workload = en_data.get("workload", exp_dir)
            world_size = en_data.get("world_size", "?")

            # Use aggregate stats if available
            dis_time = dis_data.get("total_time", {})
            en_time = en_data.get("total_time", {})
            val_time = en_data.get("validation_time", {})

            if isinstance(dis_time, dict):
                dis_mean = dis_time.get("mean", 0)
            else:
                dis_mean = dis_data.get("runtime", 0)

            if isinstance(en_time, dict):
                en_mean = en_time.get("mean", 0)
            else:
                en_mean = en_data.get("runtime", 0)

            if isinstance(val_time, dict):
                val_mean = val_time.get("mean", 0)
                val_std = val_time.get("std", 0)
            else:
                val_mean = en_data.get("validation_time", 0)
                val_std = 0

            overhead_pct = (en_mean - dis_mean) / dis_mean * 100 if dis_mean > 0 else 0
            # Propagate std as percentage
            overhead_std_pct = val_std / dis_mean * 100 if dis_mean > 0 else 0

            lines.append(
                f"| {workload} | {world_size} | {dis_mean:.4f} | {en_mean:.4f} "
                f"| {val_mean:.4f} | {overhead_pct:.2f} | {overhead_std_pct:.2f} |"
            )

    return "\n".join(lines)


# =============================================================================
# Table D — Scalability
# =============================================================================

def generate_scalability_table(bench_dir: str) -> str:
    """
    | World Size | State Size | Validation ms | Overhead % |
    """
    lines = []
    lines.append("## Table D — Scalability")
    lines.append("")
    lines.append("| World Size | Workload | Validation (ms) | Overhead % |")
    lines.append("|---:|---|---:|---:|")

    if not os.path.exists(bench_dir):
        lines.append("| *(No data)* | - | - | - |")
        return "\n".join(lines)

    entries = []
    for exp_dir in sorted(os.listdir(bench_dir)):
        full_dir = os.path.join(bench_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue

        en_path = os.path.join(full_dir, "benchmark_metrics_enabled.json")
        dis_path = os.path.join(full_dir, "benchmark_metrics_disabled.json")

        if os.path.exists(en_path):
            en_data = _load_json_safe(en_path)
            dis_data = _load_json_safe(dis_path) if os.path.exists(dis_path) else None

            if not en_data:
                continue

            world_size = en_data.get("world_size", "?")
            workload = en_data.get("workload", exp_dir)

            val_time = en_data.get("validation_time", {})
            if isinstance(val_time, dict):
                val_ms = val_time.get("mean", 0) * 1000
            else:
                val_ms = val_time * 1000

            dis_mean = 0
            en_mean = 0
            if dis_data:
                dt = dis_data.get("total_time", {})
                dis_mean = dt.get("mean", 0) if isinstance(dt, dict) else dis_data.get("runtime", 0)
            et = en_data.get("total_time", {})
            en_mean = et.get("mean", 0) if isinstance(et, dict) else en_data.get("runtime", 0)

            overhead_pct = (en_mean - dis_mean) / dis_mean * 100 if dis_mean > 0 else 0

            entries.append((world_size, workload, val_ms, overhead_pct))

    for ws, wl, vm, op in sorted(entries):
        lines.append(f"| {ws} | {wl} | {vm:.2f} | {op:.2f} |")

    return "\n".join(lines)


# =============================================================================
# Table E — Reliability
# =============================================================================

def generate_reliability_table(raw_dir: str) -> str:
    """
    | Workload | Fault | Trials | Detected | Missed | Detection Rate |
    """
    lines = []
    lines.append("## Table E — Reliability")
    lines.append("")
    lines.append("| Workload | Fault | Trials | Detected | Missed | Detection Rate |")
    lines.append("|---|---|---:|---:|---:|---:|")

    if not os.path.exists(raw_dir):
        lines.append("| *(No data)* | - | - | - | - | - |")
        return "\n".join(lines)

    # Group by (workload, fault)
    groups: Dict[tuple, List[Dict]] = {}
    for result_path in _find_result_files(raw_dir):
        data = _load_json_safe(result_path)
        if data is None or data.get("fault", "none") == "none":
            continue
        key = (data.get("workload", "?"), data.get("fault", "?"))
        groups.setdefault(key, []).append(data)

    for (workload, fault), results in sorted(groups.items()):
        n = len(results)
        detected = sum(1 for r in results if r.get("detected", False))
        missed = n - detected
        rate = (detected / n * 100) if n > 0 else 0
        lines.append(f"| {workload} | {fault} | {n} | {detected} | {missed} | {rate:.1f}% |")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    bench_dir = "results/benchmarks"
    raw_dir = "results/raw"

    report = []
    report.append("# Validation Harness Analysis Report\n")
    report.append(generate_fault_detection_table(raw_dir))
    report.append("")
    report.append(generate_resume_fidelity_table(raw_dir))
    report.append("")
    report.append(generate_overhead_table(bench_dir))
    report.append("")
    report.append(generate_scalability_table(bench_dir))
    report.append("")
    report.append(generate_reliability_table(raw_dir))
    report.append("")

    full_report = "\n".join(report)
    print(full_report)

    # Also save to file
    os.makedirs("results/analysis", exist_ok=True)
    with open("results/analysis/paper_tables.md", "w") as f:
        f.write(full_report)


if __name__ == "__main__":
    main()
