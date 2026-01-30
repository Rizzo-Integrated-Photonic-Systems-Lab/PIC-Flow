#!/usr/bin/env bash
#SBATCH --job-name=rayfield-train
#SBATCH --partition=h200_preemptable
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/rayfield_%j.out
#SBATCH --error=logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs

# -----------------------
# Copy data to local scratch for faster I/O (optional but recommended)
# -----------------------
USE_LOCAL_SCRATCH=true
ORIGINAL_DATA="/dartfs/rc/lab/R/RizzoA/f0071mj/Data_preprocessed/unified_sweep"

if [ "$USE_LOCAL_SCRATCH" = true ]; then
    LOCAL_SCRATCH="/tmp/rayfield_data_${SLURM_JOB_ID}"
    echo "Copying data to local scratch: ${LOCAL_SCRATCH}"

    # Create directory
    mkdir -p "${LOCAL_SCRATCH}"

    # Copy preprocessed data (ORIGINAL_DATA already points to unified_sweep dir)
    echo "Copying preprocessed data..."
    rsync -a --info=progress2 "${ORIGINAL_DATA}/" "${LOCAL_SCRATCH}/"

    DATA_ROOT="${LOCAL_SCRATCH}"
    echo "Data copied. Using local scratch: ${DATA_ROOT}"

    # Cleanup on exit
    trap "echo 'Cleaning up local scratch...'; rm -rf ${LOCAL_SCRATCH}" EXIT
else
    DATA_ROOT="${ORIGINAL_DATA}"
fi

# -----------------------
# Run config (adaptive weighting + curriculum gates)
# -----------------------
VERSION="unified_complexphys_v6"

# Model capacity - H200 has 80GB HBM3, can handle much larger models
HIDDEN_SIZE=16

# Data - preprocessed .pt files with pre-computed train/val/test splits
# DATA_ROOT is set above (either local scratch or original)
USE_FAST_DATASET="--use-fast-dataset"          # use FastFDTDDataset for preprocessed .pt files
USE_INDEX_SPLIT="--use-index-split"            # use pre-computed splits from index.json
# Below are only used if not using fast dataset (kept for reference)
SWEEPS="unified_sweep"                         # single unified sweep directory
USE_SHARDS="--use-shards"                      # set to "--no-use-shards" to read raw folders
SHARD_SUBDIR="."
SHARD_INDEX_NAME="index.json"
# Note: subset selection is not used when use_index_split is enabled
SUBSET_TRAIN=""
SUBSET_VAL=""
SUBSET_SEED=0

# Train
EPOCHS=750
LR=1e-4         # Lower LR for larger model stability
BATCH_SIZE=96   # H200 can handle larger batches

# Curriculum boundaries
# Phase A (epochs 1-50): FM only - learn basic field structure
# Phase B (epochs 51-150): FM + residual - enforce Helmholtz physics
# Phase C (epochs 151+): FM + residual + phase + sparam - full optimization
PHASE_A_EPOCHS=50
PHASE_B_EPOCHS=150

# Physics-informed loss weights (curriculum: losses ramp in at different phases)
# FM loss is implicit (weight=1.0). These are the auxiliary physics losses:
# - Residual: Helmholtz equation compliance (primary physics constraint)
# - Phase: accurate phase field prediction
# - S-param: device scattering parameter accuracy (modal overlap method)
# - Endpoint: denoising endpoint consistency (optional, helps early training)
# - Phase-grad: smooth phase gradients (optional, improves phase quality)
LAMBDA_ENDPOINT=0.3
LAMBDA_RESIDUAL=1.0
LAMBDA_SPARAM=1.0
LAMBDA_PHASE=0.1
LAMBDA_PHASE_GRAD=0.01
SPARAM_MODE="head"  # Use learned head with phase gradient features (fixes S11=1.0 bug)

# Warmups / start phases (control *when* tasks turn on, not their final magnitudes)
# Residual: starts Phase B with 50-epoch warmup ramp
RESIDUAL_WARMUP=50
RESIDUAL_START_PHASE="B"

# Phase/S-param/Phase-grad: start Phase C with warmups
PHASE_WARMUP=50
PHASE_START_PHASE="C"
SPARAM_WARMUP=50
SPARAM_START_PHASE="C"
PHASE_GRAD_WARMUP=50
PHASE_GRAD_START_PHASE="C"

# Spatial focus knobs
DEVICE_FOCUS_MAX=3
DEVICE_FOCUS_WARMUP=25
DEVICE_FOCUS_HOLD=200
DEVICE_FOCUS_DECAY=500
W_MIN=0.5
EPS_THR=2.5
DILATE=15

# System/env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

echo "Starting training job: ${VERSION}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NPROC_PER_NODE} WorldGPU=$((SLURM_NNODES*NPROC_PER_NODE))"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="$SLURM_JOB_ID" \
  /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  ${USE_FAST_DATASET} \
  ${USE_INDEX_SPLIT} \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --train-fraction 0.9 \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --train-num-workers 12 \
  --val-num-workers 4 \
  --prefetch-factor 8 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-warmup-epochs 50 \
  --lambda-phase "${LAMBDA_PHASE}" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode "${SPARAM_MODE}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "${RESIDUAL_START_PHASE}" \
  \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --phase-start-phase "${PHASE_START_PHASE}" \
  \
  --sparam-warmup-epochs "${SPARAM_WARMUP}" \
  --sparam-start-phase "${SPARAM_START_PHASE}" \
  \
  --phase-grad-warmup-epochs "${PHASE_GRAD_WARMUP}" \
  --phase-grad-start-phase "${PHASE_GRAD_START_PHASE}" \
  \
  --device-focus-max "${DEVICE_FOCUS_MAX}" \
  --device-focus-warmup-epochs "${DEVICE_FOCUS_WARMUP}" \
  --device-focus-hold-epochs "${DEVICE_FOCUS_HOLD}" \
  --device-focus-decay-epochs "${DEVICE_FOCUS_DECAY}" \
  --w-min "${W_MIN}" \
  --eps-thr "${EPS_THR}" \
  --dilate "${DILATE}" \
  \
  --ckpt-every 100 \
  --eval-every 15 \
  --val-batches 16 \
  \
  --pml-cells 0 \
  --physics-features \
  --complex-unet \
  --no-attention \
  --dx 0.0556 \
  \
  --warmup-epochs 10 \
  --min-lr 5e-6 \
  --log-every 1 \
  \
  --no-include-sdf \
  --use-checkpoint \
  \
  --phase-amp-tau 0.4 \
  \
  --tqdm \
  --tqdm-mininterval 15.0 \
  \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project PhotonFlow \
  \
  --no-config

echo "Training job completed."
