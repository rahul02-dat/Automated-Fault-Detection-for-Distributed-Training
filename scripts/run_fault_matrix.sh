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
    # Offline mutation
    torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_fault --config "$config"
    
    # Extract the fault name and out dir from the yaml
    FAULT_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['fault'])")
    RESUME_STEP=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['training']['resume_steps'][0])")
    EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$config')).get('experiment_id', 'run'))")
    OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$config')).get('output_dir', f'results/raw/{EXP_ID}'))")
    
    echo "Mutated checkpoint for $FAULT_NAME created."
    
    # Now run resume
    CKPT_PATH="${OUT_DIR}/checkpoint_${RESUME_STEP}_mutated_${FAULT_NAME}"
    echo "Resuming from mutated checkpoint: $CKPT_PATH"
    
    torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=2 -m src.experiments.run_resume --config "$config" --ckpt-path "$CKPT_PATH"
done

echo "Fault Matrix Completed!"
