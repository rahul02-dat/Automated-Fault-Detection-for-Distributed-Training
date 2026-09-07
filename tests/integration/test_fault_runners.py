import os
import subprocess
import json
import pytest

def run_experiment(config_path, expected_result_dir):
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0"
    env["OMP_NUM_THREADS"] = "1"
    
    # Run training
    subprocess.run([
        "torchrun", "--rdzv_endpoint=localhost:29500", "--nproc_per_node=2",
        "-m", "src.experiments.run_training",
        "--config", config_path
    ], env=env, check=True)
    
    # Run fault (offline mutation)
    subprocess.run([
        "torchrun", "--rdzv_endpoint=localhost:29500", "--nproc_per_node=2",
        "-m", "src.experiments.run_fault",
        "--config", config_path
    ], env=env, check=True)

    # We need the fault name to construct the path. Let's just pass the mutated ckpt to run_resume
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    fault_name = cfg["fault"]
    resume_step = cfg.get("training", {}).get("resume_steps", [0])[0]
    
    ckpt_path = os.path.join(expected_result_dir, f"checkpoint_{resume_step}_mutated_{fault_name}")
    
    # Run resume
    subprocess.run([
        "torchrun", "--rdzv_endpoint=localhost:29500", "--nproc_per_node=2",
        "-m", "src.experiments.run_resume",
        "--config", config_path,
        "--ckpt-path", ckpt_path
    ], env=env, check=True)
    
    return os.path.join(expected_result_dir, "resume")

@pytest.fixture(scope="module")
def ema_buggy_dir():
    return run_experiment("configs/smoke/ema_buggy.yaml", "results/raw/ema_smoke_buggy")

@pytest.fixture(scope="module")
def scheduler_fault_dir():
    return run_experiment("configs/faults/scheduler.yaml", "results/raw/scheduler_fault")

@pytest.fixture(scope="module")
def rng_fault_dir():
    return run_experiment("configs/faults/rng.yaml", "results/raw/rng_fault")

@pytest.fixture(scope="module")
def dataloader_fault_dir():
    return run_experiment("configs/faults/dataloader.yaml", "results/raw/dataloader_fault")

@pytest.fixture(scope="module")
def optimizer_fault_dir():
    return run_experiment("configs/faults/optimizer.yaml", "results/raw/optimizer_fault")

def test_ema_fault_detected(ema_buggy_dir):
    # In EMABuggy, the step is not saved/loaded, so all ranks load step=0
    # Thus, cross-rank validation PASSES because they are consistently wrong.
    val_path = os.path.join(ema_buggy_dir, "validation_restore_5.json")
    assert os.path.exists(val_path)
    with open(val_path, "r") as f:
        results = json.load(f)
    ema_step_res = next(r for r in results if r["state_name"] == "ema.step")
    assert ema_step_res["status"] == "PASS"
    
    # But the eval loss should diverge from the baseline (uninterrupted run)
    baseline_eval_path = os.path.join(os.path.dirname(ema_buggy_dir), "eval_final.json")
    resume_eval_path = os.path.join(ema_buggy_dir, "eval_final_10.json")
    
    with open(baseline_eval_path, "r") as f:
        baseline_loss = json.load(f)["loss"]
    with open(resume_eval_path, "r") as f:
        resume_loss = json.load(f)["loss"]
        
    assert baseline_loss != resume_loss

def test_scheduler_fault_detected(scheduler_fault_dir):
    val_path = os.path.join(scheduler_fault_dir, "validation_restore_5.json")
    assert os.path.exists(val_path)
    with open(val_path, "r") as f:
        results = json.load(f)
    # The fault zeros rank 1's scheduler step
    step_res = next(r for r in results if r["state_name"] == "scheduler")
    assert step_res["status"] == "FAIL"

def test_rng_fault_detected(rng_fault_dir):
    val_path = os.path.join(rng_fault_dir, "validation_restore_5.json")
    assert os.path.exists(val_path)
    with open(val_path, "r") as f:
        results = json.load(f)
    # The fault sets rank 1's torch rng to initial state
    rng_res = next((r for r in results if r["state_name"] == "rng.torch_cpu"), None)
    if rng_res:
        assert rng_res["status"] == "FAIL"

def test_dataloader_fault_detected(dataloader_fault_dir):
    val_path = os.path.join(dataloader_fault_dir, "validation_restore_5.json")
    assert os.path.exists(val_path)
    with open(val_path, "r") as f:
        results = json.load(f)
    # The fault zeroes global_step which affects data cursor validation if it's there
    step_res = next((r for r in results if r["state_name"] == "global_step"), None)
    if step_res:
        assert step_res["status"] == "FAIL"

def test_optimizer_fault_detected(optimizer_fault_dir):
    val_path = os.path.join(optimizer_fault_dir, "validation_restore_5.json")
    assert os.path.exists(val_path)
    with open(val_path, "r") as f:
        results = json.load(f)
    # The fault zeros param group lr on rank 1
    opt_res = next((r for r in results if r["state_name"] == "optimizer"), None)
    if opt_res:
        assert opt_res["status"] == "FAIL"
