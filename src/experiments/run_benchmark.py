import argparse
import os
import yaml
import json
import subprocess
import time
from pathlib import Path

def run_subprocess(cmd, env=None):
    env = os.environ.copy() if env is None else env
    env["GLOO_SOCKET_IFNAME"] = "lo0"
    env["OMP_NUM_THREADS"] = "1"
    
    start_time = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    duration = time.time() - start_time
    return result, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Running benchmark suite from {args.config}")
    
    benchmark_id = config.get("benchmark_id", "bench_run")
    out_dir = f"results/processed/{benchmark_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    report = {
        "benchmark_id": benchmark_id,
        "results": []
    }

    world_size = config.get("world_size", 2)
    faults = config.get("faults", [])
    
    # Run Baseline (healthy)
    baseline_config_path = config.get("baseline_config", "configs/smoke/ema_fixed.yaml")
    print(f"Running baseline training: {baseline_config_path}")
    
    cmd_train = [
        "torchrun", 
        "--rdzv_endpoint=localhost:29500", 
        f"--nproc_per_node={world_size}", 
        "-m", "src.experiments.run_training", 
        "--config", baseline_config_path
    ]
    
    res, baseline_dur = run_subprocess(cmd_train)
    if res.returncode != 0:
        print(f"Baseline training failed: {res.stderr}")
        return
        
    print(f"Baseline completed in {baseline_dur:.2f}s")

    for fault_cfg_path in faults:
        print(f"Running fault injection: {fault_cfg_path}")
        cmd_fault = [
            "torchrun", 
            "--rdzv_endpoint=localhost:29500", 
            f"--nproc_per_node={world_size}", 
            "-m", "src.experiments.run_fault", 
            "--config", fault_cfg_path
        ]
        
        res, fault_dur = run_subprocess(cmd_fault)
        
        # We also need to extract validation results
        with open(fault_cfg_path, "r") as f:
            fault_cfg = yaml.safe_load(f)
            
        exp_id = fault_cfg.get("experiment_id", "run")
        raw_dir = f"results/raw/{exp_id}/fault_run"
        
        # Find validation files
        detected = False
        val_ms = 0
        
        try:
            val_files = [f for f in os.listdir(raw_dir) if f.startswith("validation_")]
            for vf in val_files:
                with open(os.path.join(raw_dir, vf), "r") as f:
                    val_data = json.load(f)
                    for item in val_data:
                        val_ms += item.get("duration_ms", 0)
                        if item.get("status") == "FAIL":
                            detected = True
        except Exception as e:
            print(f"Failed to parse validation results for {fault_cfg_path}: {e}")
            
        overhead_pct = ((fault_dur - baseline_dur) / baseline_dur) * 100 if baseline_dur > 0 else 0
        
        fault_result = {
            "config": fault_cfg_path,
            "detected": detected,
            "fault_dur_s": fault_dur,
            "baseline_dur_s": baseline_dur,
            "overhead_pct": overhead_pct,
            "validation_ms": val_ms
        }
        report["results"].append(fault_result)
        
        print(f"  Detected: {detected} | Overhead: {overhead_pct:.2f}% | Val time: {val_ms:.2f}ms")

    report_path = os.path.join(out_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Benchmark completed. Report saved to {report_path}")

if __name__ == "__main__":
    main()
