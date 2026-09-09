#!/usr/bin/env bash
set -e

# Run the complete fault matrix.
# Does NOT use `|| true` to hide failures.
# The resume runner uses ExperimentOutcome semantics to distinguish
# EXPECTED_DETECTION from UNEXPECTED_PASS/UNEXPECTED_FAILURE.

WORKLOADS=("tiny_transformer" "resnet" "small_transformer")
FAULTS=("ema_scalar_omission" "scheduler_stale_state" "rng_state_omission" "data_cursor_mismatch" "optimizer_state_corruption")
REPETITIONS=${1:-1}

export GLOO_SOCKET_IFNAME="lo0"
export OMP_NUM_THREADS="1"
export TORCH_CPP_LOG_LEVEL=ERROR

echo "=============================================="
echo "Starting Full Fault Matrix (${REPETITIONS} reps)"
echo "=============================================="

MATRIX_STATUS=0
MATRIX_SUMMARY=""

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
        for REP in $(seq 1 $REPETITIONS); do
            echo "----------------------------------------------"
            echo "Testing Fault: $FAULT on Workload: $WORKLOAD (rep $REP/$REPETITIONS)"
            echo "----------------------------------------------"

            # We need a temporary config with the fault injected
            TMP_CONFIG="/tmp/matrix_config_$$.yaml"
            sed "s/fault: none/fault: $FAULT/" $CONFIG > $TMP_CONFIG

            # Inject the fault into the baseline checkpoint
            uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29500 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_fault --config $TMP_CONFIG

            # Resume and validate — capture exit code without suppressing it
            RESUME_EXIT=0
            uv run torchrun --local-addr=127.0.0.1 --rdzv_endpoint=127.0.0.1:29500 --standalone --nnodes=1 --nproc_per_node=2 -m src.experiments.run_resume --config $TMP_CONFIG || RESUME_EXIT=$?

            if [ $RESUME_EXIT -ne 0 ]; then
                echo "[MATRIX] $WORKLOAD/$FAULT rep=$REP: Resume exited with code $RESUME_EXIT"
                MATRIX_SUMMARY="${MATRIX_SUMMARY}\n${WORKLOAD}/${FAULT}/rep${REP}: EXIT_CODE=${RESUME_EXIT}"
            else
                MATRIX_SUMMARY="${MATRIX_SUMMARY}\n${WORKLOAD}/${FAULT}/rep${REP}: COMPLETED"
            fi

            rm -f $TMP_CONFIG
        done
    done
done

echo ""
echo "=============================================="
echo "Matrix Complete!"
echo "=============================================="
echo -e "Summary:${MATRIX_SUMMARY}"
echo ""
echo "Run ./scripts/generate_paper_tables.sh to generate analysis tables."
