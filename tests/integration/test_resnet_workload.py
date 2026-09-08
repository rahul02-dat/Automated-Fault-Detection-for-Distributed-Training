import os
import pytest
import subprocess
import yaml
import shutil
import json

@pytest.fixture
def resnet_config(tmp_path):
    config = {
        "experiment_id": "test_resnet",
        "output_dir": str(tmp_path),
        "workload": {
            "name": "resnet",
            "config": {
                "batch_size": 4,
                "lr": 0.01
            }
        },
        "distributed": {
            "world_size": 2,
            "backend": "gloo"
        },
        "training": {
            "seed": 42,
            "total_steps": 2,
            "checkpoint_steps": [1],
            "resume_steps": [1]
        },
        "fault": "none"
    }
    
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
        
    return cfg_path, tmp_path

def test_resnet_training_and_resume(resnet_config):
    cfg_path, tmp_path = resnet_config
    
    # 1. Run Baseline Training
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0"
    env["OMP_NUM_THREADS"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29501"
    
    cmd_train = [
        "torchrun",
        "--rdzv_endpoint=localhost:29501",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m", "src.experiments.run_training",
        "--config", str(cfg_path)
    ]
    
    result_train = subprocess.run(cmd_train, env=env, capture_output=True, text=True)
    assert result_train.returncode == 0, f"Training failed:\n{result_train.stderr}"
    
    # Check outputs
    assert os.path.exists(tmp_path / "checkpoint_1")
    assert os.path.exists(tmp_path / "checkpoint_final")
    
    # 2. Run Resume
    cmd_resume = [
        "torchrun",
        "--rdzv_endpoint=localhost:29501",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m", "src.experiments.run_resume",
        "--config", str(cfg_path),
        "--strict"
    ]
    
    result_resume = subprocess.run(cmd_resume, env=env, capture_output=True, text=True)
    assert result_resume.returncode == 0, f"Resume failed:\n{result_resume.stderr}"
    
    # Check outputs
    resume_dir = tmp_path / "resume"
    assert os.path.exists(resume_dir / "checkpoint_final")
