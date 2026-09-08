import os
import json
import glob

def analyze_benchmarks(bench_dir):
    print("==============================================")
    print("Table 1: Cross-Rank Validation Overhead")
    print("==============================================")
    print("| Workload | Disabled Step Time (s) | Enabled Step Time (s) | Validation Overhead (s) | Overhead % |")
    print("|----------|------------------------|-----------------------|-------------------------|------------|")
    
    if not os.path.exists(bench_dir):
        print("| (No benchmark data found) | - | - | - | - |")
        return

    workloads = []
    for exp_dir in os.listdir(bench_dir):
        full_dir = os.path.join(bench_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue
            
        dis_path = os.path.join(full_dir, "benchmark_metrics_disabled.json")
        en_path = os.path.join(full_dir, "benchmark_metrics_enabled.json")
        
        if os.path.exists(dis_path) and os.path.exists(en_path):
            with open(dis_path) as f:
                dis_data = json.load(f)
            with open(en_path) as f:
                en_data = json.load(f)
                
            dis_total_time = dis_data.get("runtime", 0) - dis_data.get("checkpoint_time", 0)
            dis_step = dis_total_time / max(1, dis_data.get("total_steps", 1))
            
            en_total_time = en_data.get("runtime", 0) - en_data.get("checkpoint_time", 0)
            en_step = en_total_time / max(1, en_data.get("total_steps", 1))
            
            val_overhead_per_checkpoint = en_data.get("validation_time", 0)
            overhead_pct = (val_overhead_per_checkpoint / dis_total_time) * 100 if dis_total_time > 0 else 0
            
            workloads.append((exp_dir, dis_step, en_step, val_overhead_per_checkpoint, overhead_pct))
            
    for w, ds, es, vo, op in sorted(workloads):
        print(f"| {w:20s} | {ds:.4f} | {es:.4f} | {vo:.4f} | {op:.2f}% |")
    print("\n")


def analyze_fault_matrix(raw_dir):
    print("==============================================")
    print("Table 2: Fault Detection Matrix")
    print("==============================================")
    
    if not os.path.exists(raw_dir):
        print("(No raw data found)")
        return
        
    print("| Experiment | Phase | Faults Detected by Contracts | Overall Status |")
    print("|------------|-------|------------------------------|----------------|")
    
    for exp_dir in sorted(os.listdir(raw_dir)):
        full_dir = os.path.join(raw_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue
            
        val_files = glob.glob(os.path.join(full_dir, "**", "validation_*.json"), recursive=True)
        for vf in val_files:
            rel_path = os.path.relpath(vf, raw_dir)
            
            with open(vf) as f:
                try:
                    data = json.load(f)
                except:
                    continue
                    
            if not isinstance(data, list):
                continue
                
            failed_contracts = [d["state_name"] for d in data if d.get("status") == "FAIL"]
            overall_status = "FAIL" if failed_contracts else "PASS"
            
            contracts_str = ", ".join(failed_contracts) if failed_contracts else "None"
            
            phase = os.path.basename(vf).replace(".json", "").replace("validation_", "")
            
            print(f"| {exp_dir} | {phase} | {contracts_str} | {overall_status} |")
    print("\n")


def analyze_recovery_fidelity(raw_dir):
    print("==============================================")
    print("Table 3: End-to-End Recovery Fidelity")
    print("==============================================")
    
    if not os.path.exists(raw_dir):
        print("(No raw data found)")
        return
        
    print("| Experiment | Baseline Loss | Resumed Loss | Diff | Exact Match? |")
    print("|------------|---------------|--------------|------|--------------|")
    
    for exp_dir in sorted(os.listdir(raw_dir)):
        full_dir = os.path.join(raw_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue
            
        eval_files = glob.glob(os.path.join(full_dir, "**", "eval_*.json"), recursive=True)
        if not eval_files:
            continue
            
        losses = []
        for ef in eval_files:
            with open(ef) as f:
                try:
                    data = json.load(f)
                    losses.append(data.get("loss", 0.0))
                except:
                    pass
        
        if len(losses) >= 1:
            baseline = losses[0]
            resumed = losses[-1]
            diff = abs(baseline - resumed)
            exact = "Yes" if diff < 1e-5 else "No"
            
            print(f"| {exp_dir} | {baseline:.6f} | {resumed:.6f} | {diff:.6f} | {exact} |")
            
    print("\n")


def main():
    bench_dir = "results/benchmarks"
    raw_dir = "results/raw"
    
    print("# Validation Harness Analysis Report\n")
    analyze_benchmarks(bench_dir)
    analyze_fault_matrix(raw_dir)
    analyze_recovery_fidelity(raw_dir)

if __name__ == "__main__":
    main()
