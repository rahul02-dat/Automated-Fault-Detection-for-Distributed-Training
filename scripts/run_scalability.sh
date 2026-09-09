#!/usr/bin/env bash
set -e

# Scalability runner — measures validation overhead across world sizes.
# Targets 1, 2, 4 ranks (feasible on most machines).

REPETITIONS=${1:-5}
WORLD_SIZES="1 2"

# Check GPU count for 4-rank support
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    if [ "$GPU_COUNT" -ge 4 ]; then
        WORLD_SIZES="1 2 4"
    fi
fi

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

    for WS in $WORLD_SIZES; do
        echo "=============================================="
        echo "Scalability: $WORKLOAD, world_size=$WS (${REPETITIONS} reps)"
        echo "=============================================="

        # Create temporary config with modified world_size
        TMP_CONFIG="/tmp/scalability_config_$$.yaml"
        sed "s/world_size: [0-9]*/world_size: $WS/" $CONFIG > $TMP_CONFIG
        # Also update the experiment_id to include world_size
        sed -i '' "s/experiment_id: .*/experiment_id: ${WORKLOAD}_ws${WS}/" $TMP_CONFIG

        echo "--- Validation DISABLED ---"
        uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29510 --standalone --nnodes=1 --nproc_per_node=$WS \
            -m src.experiments.run_benchmark --config $TMP_CONFIG --disable-validation --repetitions $REPETITIONS

        echo "--- Validation ENABLED ---"
        uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29511 --standalone --nnodes=1 --nproc_per_node=$WS \
            -m src.experiments.run_benchmark --config $TMP_CONFIG --repetitions $REPETITIONS

        rm -f $TMP_CONFIG
    done
done

echo ""
echo "Scalability tests complete! Check results/benchmarks/"
