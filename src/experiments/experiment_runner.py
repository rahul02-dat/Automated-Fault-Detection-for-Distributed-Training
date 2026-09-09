"""
Experiment runner orchestrator (P6).

Aggregates results from repeated trial runs and computes:
- TP, FN, FP, errors, unknowns
- TPR/recall, FPR, detection rate
- Healthy-restart false positive rate
"""
import os
import json
import glob
from typing import Any, Dict, List

from .result_schema import ExperimentResult, ExperimentOutcome


def aggregate_fault_results(raw_dir: str) -> Dict[str, Any]:
    """
    Aggregate experiment results from the raw directory.

    Returns a dict keyed by (workload, fault) with aggregate statistics.
    """
    groups: Dict[tuple, List[ExperimentResult]] = {}

    for result_path in glob.glob(os.path.join(raw_dir, "**", "raw_results.json"), recursive=True):
        try:
            result = ExperimentResult.load(result_path)
        except Exception as e:
            print(f"Warning: Could not load {result_path}: {e}")
            continue

        key = (result.workload, result.fault)
        groups.setdefault(key, []).append(result)

    aggregated = {}
    for (workload, fault), results in sorted(groups.items()):
        n = len(results)

        if fault == "none":
            # Healthy runs — check for false positives
            fp = sum(1 for r in results if r.detected)
            tn = n - fp
            aggregated[(workload, fault)] = {
                "workload": workload,
                "fault": fault,
                "trials": n,
                "true_negatives": tn,
                "false_positives": fp,
                "false_positive_rate": fp / n if n > 0 else 0.0,
            }
        else:
            # Fault runs — check for detection
            tp = sum(1 for r in results if r.outcome == ExperimentOutcome.EXPECTED_DETECTION.value)
            fn = sum(1 for r in results if r.outcome == ExperimentOutcome.UNEXPECTED_PASS.value)
            errors = sum(1 for r in results if r.outcome == ExperimentOutcome.EXPERIMENT_ERROR.value)
            unknowns = n - tp - fn - errors

            # Causal attribution
            attribution_pass = sum(1 for r in results if r.causal_attribution == "PASS")
            attribution_fail = sum(1 for r in results if r.causal_attribution == "FAIL")

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            aggregated[(workload, fault)] = {
                "workload": workload,
                "fault": fault,
                "trials": n,
                "true_positives": tp,
                "false_negatives": fn,
                "errors": errors,
                "unknowns": unknowns,
                "true_positive_rate": tpr,
                "detection_rate": tp / n if n > 0 else 0.0,
                "attribution_pass": attribution_pass,
                "attribution_fail": attribution_fail,
            }

    return aggregated


def generate_aggregate_report(raw_dir: str, output_path: str = None) -> str:
    """
    Generate a human-readable aggregate report.
    """
    aggregated = aggregate_fault_results(raw_dir)

    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT AGGREGATE REPORT")
    lines.append("=" * 80)

    # Healthy runs
    healthy = {k: v for k, v in aggregated.items() if k[1] == "none"}
    if healthy:
        lines.append("\n--- Healthy Runs (False Positive Check) ---")
        for key, stats in sorted(healthy.items()):
            lines.append(
                f"  {stats['workload']:25s} | "
                f"Trials={stats['trials']:3d} | "
                f"TN={stats['true_negatives']:3d} | "
                f"FP={stats['false_positives']:3d} | "
                f"FPR={stats['false_positive_rate']:.1%}"
            )

    # Fault runs
    faults = {k: v for k, v in aggregated.items() if k[1] != "none"}
    if faults:
        lines.append("\n--- Fault Detection ---")
        for key, stats in sorted(faults.items()):
            lines.append(
                f"  {stats['workload']:25s} | {stats['fault']:30s} | "
                f"Trials={stats['trials']:3d} | "
                f"TP={stats['true_positives']:3d} | "
                f"FN={stats['false_negatives']:3d} | "
                f"ERR={stats['errors']:3d} | "
                f"TPR={stats['true_positive_rate']:.1%} | "
                f"Attr PASS={stats['attribution_pass']:3d}"
            )

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

        # Also save as JSON
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump(
                {str(k): v for k, v in aggregated.items()},
                f, indent=2
            )

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--raw-dir", default="results/raw", help="Raw results directory")
    parser.add_argument("--output", default="results/analysis/aggregate_report.txt", help="Output report path")
    args = parser.parse_args()

    generate_aggregate_report(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
