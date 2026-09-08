#!/usr/bin/env bash
set -e

# Benchmark runner wrapper

if [ -z "$1" ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG=$1

echo "=============================================="
echo "Running Benchmark with Validation DISABLED"
echo "=============================================="
uv run torchrun --rdzv_endpoint=localhost:29505 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_benchmark --config $CONFIG --disable-validation

echo "=============================================="
echo "Running Benchmark with Validation ENABLED"
echo "=============================================="
uv run torchrun --rdzv_endpoint=localhost:29506 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_benchmark --config $CONFIG

echo "Benchmark complete! Check results/benchmarks/ directory."
