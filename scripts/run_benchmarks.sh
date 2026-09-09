#!/usr/bin/env bash
set -e

# Benchmark runner — measures validation overhead across workloads.
# Runs validation disabled, then enabled, with multiple repetitions.

REPETITIONS=${1:-5}

WORKLOADS=("tiny_transformer" "resnet" "small_transformer")

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"
export TORCH_CPP_LOG_LEVEL=ERROR

for WORKLOAD in "${WORKLOADS[@]}"; do
    CONFIG="configs/benchmarks/${WORKLOAD}_smoke.yaml"
    if [ ! -f "$CONFIG" ]; then
        if [ "$WORKLOAD" == "tiny_transformer" ]; then
            CONFIG="configs/smoke/ema_fixed.yaml"
        else
            echo "Skipping $WORKLOAD: Config not found."
            continue
        fi
    fi

    echo "=============================================="
    echo "Benchmarking $WORKLOAD (${REPETITIONS} reps)"
    echo "=============================================="

    echo "--- Validation DISABLED ---"
    uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29505 --standalone --nnodes=1 --nproc_per_node=2 \
        -m src.experiments.run_benchmark --config $CONFIG --disable-validation --repetitions $REPETITIONS

    echo "--- Validation ENABLED ---"
    uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29506 --standalone --nnodes=1 --nproc_per_node=2 \
        -m src.experiments.run_benchmark --config $CONFIG --repetitions $REPETITIONS
done

echo ""
echo "Benchmark complete! Check results/benchmarks/ directory."
echo "Run ./scripts/generate_paper_tables.sh to generate analysis tables."
