#!/bin/bash
set -e

echo "Running Full Fault Matrix..."

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"

source .venv/bin/activate

echo "Running baseline to generate checkpoints for faults..."
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_training --config configs/smoke/ema_fixed.yaml

for config in configs/faults/*.yaml; do
    echo "Running fault injection with config: $config"
    torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_fault --config "$config"
done

echo "Fault Matrix Completed!"
