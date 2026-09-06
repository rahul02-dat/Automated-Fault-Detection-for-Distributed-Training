#!/bin/bash
set -e

echo "Running Smoke Test..."

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"

source .venv/bin/activate

echo "1. Running Baseline Training (ema_fixed)..."
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_training --config configs/smoke/ema_fixed.yaml

echo "2. Running Fault Injection (ema_buggy)..."
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_fault --config configs/smoke/ema_buggy.yaml

echo "Smoke Test Completed Successfully!"
