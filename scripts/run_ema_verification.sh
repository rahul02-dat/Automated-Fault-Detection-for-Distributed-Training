#!/bin/bash
set -e

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"

source .venv/bin/activate

echo "Running end-to-end 4-rank EMA verification..."

# 1. Run Baseline Training (generates checkpoints)
echo "=== Baseline Training ==="
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_training --config configs/smoke/ema_fixed.yaml

# 2. Run Fault Injection (offline mutation)
echo "=== Fault Injection ==="
# Copy the baseline checkpoint to the buggy experiment's directory
mkdir -p results/raw/ema_smoke_buggy
rm -rf results/raw/ema_smoke_buggy/checkpoint_5
cp -r results/raw/ema_smoke_fixed/checkpoint_5 results/raw/ema_smoke_buggy/
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_fault --config configs/smoke/ema_buggy.yaml

# 3. Run Double Resume (First from baseline, then from faulty)
# We need to manually construct the resume commands.

EXP_ID="ema_smoke_buggy"
OUT_DIR="results/raw/ema_smoke_buggy"
RESUME_STEP=5
FAULT_NAME="ema_scalar_omission"

echo "=== Resuming from Baseline Checkpoint ==="
# We can use ema_fixed.yaml to resume the baseline properly without the buggy config's run_name override
# Wait, let's just use run_resume with ema_fixed.yaml
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_resume --config configs/smoke/ema_fixed.yaml --ckpt-path "results/raw/ema_smoke_fixed/checkpoint_5"

echo "=== Resuming from Faulty Checkpoint ==="
CKPT_PATH="${OUT_DIR}/checkpoint_${RESUME_STEP}_mutated_${FAULT_NAME}"
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 -m src.experiments.run_resume --config configs/smoke/ema_buggy.yaml --ckpt-path "$CKPT_PATH"

echo "Verification complete!"
