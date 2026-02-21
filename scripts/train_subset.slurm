#!/usr/bin/env bash
#SBATCH --job-name=rayfield-5dev-a5500
#SBATCH --partition=a5500
#SBATCH --nodelist=centurion[01-05]
#SBATCH --gres=gpu:nvidia_rtx_a5500:8
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/rayfield_5dev_a5500_%j.out
#SBATCH --error=logs/rayfield_5dev_a5500_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

# ==============================================================================
# Rayfield 5-Device Subset Training
# ==============================================================================
# Devices: straight, sbend, directional_coupler, ybranch, euler_bend
# Losses:  Flow Matching + Residual + Endpoint + S-Parameter (NO phase losses)
# Dataset: ~5000 samples (4500 train / 500 val)
# ==============================================================================

source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs

# -----------------------
# Copy data to local scratch for faster I/O (optional but recommended)
# -----------------------
USE_LOCAL_SCRATCH=false  # Disabled for now - symlinks already created
ORIGINAL_DATA="/dartfs/rc/lab/R/RizzoA/f0071mj/Data/unified_sweep"

if [ "$USE_LOCAL_SCRATCH" = true ]; then
    LOCAL_SCRATCH="/tmp/rayfield_5dev_${SLURM_JOB_ID}"
    echo "Copying data to local scratch: ${LOCAL_SCRATCH}"
    mkdir -p "${LOCAL_SCRATCH}/shards"
    echo "Copying shard data..."
    rsync -a --info=progress2 "${ORIGINAL_DATA}/shards/" "${LOCAL_SCRATCH}/shards/"
    SOURCE_DATA="${LOCAL_SCRATCH}"
    echo "Data copied. Using local scratch: ${SOURCE_DATA}"
    trap "echo 'Cleaning up local scratch...'; rm -rf ${LOCAL_SCRATCH}" EXIT
else
    SOURCE_DATA="${ORIGINAL_DATA}"
fi

# -----------------------
# Run config
# -----------------------
VERSION="subset_5dev_$(date +%Y%m%d_%H%M)"

# Training parameters
EPOCHS=500
BATCH_SIZE=80        # Global batch size (divided across all GPUs)
HIDDEN_SIZE=16

# Dataset - use pre-filtered index under Data/filtered_5devices
DATA_ROOT="/dartfs/rc/lab/R/RizzoA/f0071mj/Data"
INCLUDE_SWEEPS="filtered_5devices"

# Curriculum boundaries
PHASE_A_EPOCHS=25    # FM only
PHASE_B_EPOCHS=100   # FM + Residual

# Loss weights (NO phase losses)
LAMBDA_RESIDUAL=1.0
LAMBDA_ENDPOINT=0.5
LAMBDA_SPARAM=0.5
LAMBDA_PHASE=0.0         # DISABLED
LAMBDA_PHASE_GRAD=0.0    # DISABLED

# Warmups
RESIDUAL_WARMUP=25
ENDPOINT_WARMUP=25
SPARAM_WARMUP=50

# S-param settings
SPARAM_MODE="modal"
SPARAM_EVERY=1

# Unrolling (PBFM)
UNROLL_STEPS=4

# Learning rate
LR=1e-4
WARMUP_EPOCHS=5
MIN_LR=1e-6

# Logging
LOG_EVERY=1
EVAL_EVERY=10
CKPT_EVERY=25
SAMPLE_EVAL_LIMIT=16
VAL_BATCHES=32

# System/env
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"

echo "=============================================="
echo " RAYFIELD - 5-Device Subset Training"
echo "=============================================="
echo "Start:      $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Version:    ${VERSION}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NPROC_PER_NODE} WorldGPU=$((SLURM_NNODES*NPROC_PER_NODE))"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Data root:  ${DATA_ROOT}"
echo "Sweeps:     ${INCLUDE_SWEEPS}"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Epochs:        ${EPOCHS}"
echo "  Batch size:    ${BATCH_SIZE} (global)"
echo "  Hidden size:   ${HIDDEN_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Curriculum:    Phase A (0-${PHASE_A_EPOCHS}), Phase B (${PHASE_A_EPOCHS}-${PHASE_B_EPOCHS}), Phase C (${PHASE_B_EPOCHS}+)"
echo ""
echo "Losses:"
echo "  Residual:      ${LAMBDA_RESIDUAL} (warmup ${RESIDUAL_WARMUP})"
echo "  Endpoint:      ${LAMBDA_ENDPOINT} (warmup ${ENDPOINT_WARMUP})"
echo "  S-param:       ${LAMBDA_SPARAM} (warmup ${SPARAM_WARMUP})"
echo "  Phase:         ${LAMBDA_PHASE} (DISABLED)"
echo "  Phase-grad:    ${LAMBDA_PHASE_GRAD} (DISABLED)"
echo ""
echo "Unrolling:       ${UNROLL_STEPS} steps"
echo "=============================================="

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="$SLURM_JOB_ID" \
  /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  --use-shards \
  --use-index-split \
  --include-sweeps "${INCLUDE_SWEEPS}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --use-checkpoint \
  --train-num-workers 4 \
  --val-num-workers 2 \
  --prefetch-factor 4 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode "${SPARAM_MODE}" \
  --sparam-every "${SPARAM_EVERY}" \
  --sparam-warmup-epochs "${SPARAM_WARMUP}" \
  \
  --lambda-phase "${LAMBDA_PHASE}" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  \
  --unroll-steps "${UNROLL_STEPS}" \
  \
  --log-every "${LOG_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  --sample-eval-limit "${SAMPLE_EVAL_LIMIT}" \
  --val-batches "${VAL_BATCHES}" \
  \
  --pml-cells 0 \
  --physics-features \
  --complex-unet \
  --no-attention \
  --dx 0.0556 \
  \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --min-lr "${MIN_LR}" \
  \
  --no-include-sdf \
  --use-checkpoint \
  \
  --tqdm \
  --tqdm-mininterval 15.0 \
  \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project PhotonFlow \
  \
  --no-config

echo ""
echo "=============================================="
echo " Training Complete"
echo " End: $(date)"
echo "=============================================="
