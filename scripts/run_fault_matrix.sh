#!/usr/bin/env bash
set -e

# Run the complete fault matrix

WORKLOADS=("tiny_transformer" "resnet" "small_transformer")
FAULTS=("ema_scalar_omission" "scheduler_stale_state" "rng_state_omission" "data_cursor_mismatch" "optimizer_state_corruption")

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"
export TORCH_CPP_LOG_LEVEL=ERROR

echo "=============================================="
echo "Starting Full Fault Matrix"
echo "=============================================="

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

    echo "Running baseline for $WORKLOAD..."
    uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29500 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_training --config $CONFIG

    for FAULT in "${FAULTS[@]}"; do
        echo "----------------------------------------------"
        echo "Testing Fault: $FAULT on Workload: $WORKLOAD"
        echo "----------------------------------------------"
        
        # We need a temporary config with the fault injected
        TMP_CONFIG="/tmp/matrix_config_$$.yaml"
        sed "s/fault: none/fault: $FAULT/" $CONFIG > $TMP_CONFIG
        
        # Inject the fault into the baseline checkpoint
        uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29500 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_fault --config $TMP_CONFIG
        
        # Resume and validate (we don't exit on strict here so we can finish the matrix)
        uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29500 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_resume --config $TMP_CONFIG || true
        
        rm $TMP_CONFIG
    done
done

echo "Matrix complete!"
