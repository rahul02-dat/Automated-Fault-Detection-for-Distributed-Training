#!/bin/bash
set -e

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"

source .venv/bin/activate

echo "Running end-to-end 4-rank EMA verification with double resume..."

# 1. Run Baseline Training (generates checkpoint 4 and 8)
echo "=== Baseline Training ==="
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_training --config configs/smoke/double_resume_fixed.yaml

# 2. Run Fault Injection on checkpoint 4
echo "=== Fault Injection ==="
mkdir -p results/raw/ema_double_resume_buggy
rm -rf results/raw/ema_double_resume_buggy/checkpoint_4
cp -r results/raw/ema_double_resume_fixed/checkpoint_4 results/raw/ema_double_resume_buggy/
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_fault --config configs/smoke/double_resume_buggy.yaml

# 3. First Resume (from faulty checkpoint 4 -> trains to 8)
echo "=== First Resume (Faulty) ==="
CKPT_PATH="results/raw/ema_double_resume_buggy/checkpoint_4_mutated_ema_scalar_omission"
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_resume --config configs/smoke/double_resume_buggy.yaml --ckpt-path "$CKPT_PATH"

# The first resume saves a new checkpoint at step 8 in results/raw/ema_double_resume_buggy/resume/checkpoint_8
# 4. Double Resume (from the resumed checkpoint 8 -> trains to 12)
echo "=== Double Resume (Faulty) ==="
DOUBLE_CKPT_PATH="results/raw/ema_double_resume_buggy/resume/checkpoint_8"
# We need to run resume again. The run_resume logic will output to `resume/` which will overwrite the previous results
# but we can capture the final evaluation to verify that double resume works.
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_resume --config configs/smoke/double_resume_buggy.yaml --ckpt-path "$DOUBLE_CKPT_PATH"

echo "Verification complete!"
