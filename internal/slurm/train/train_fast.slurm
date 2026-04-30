#!/usr/bin/env bash
#SBATCH --job-name=rayfield-fast
#SBATCH --partition=h200_preemptable
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/rayfield_fast_%j.out
#SBATCH --error=logs/rayfield_fast_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# -----------------------
# FAST DATASET - uses preprocessed .pt files
# Run preprocess_data.slurm first!
# -----------------------
DATA_ROOT="/dartfs/rc/lab/R/RizzoA/f0071mj/Data_preprocessed/unified_sweep"

VERSION="unified_complexphys_fast_v1"
HIDDEN_SIZE=64          # Can use larger model now
EPOCHS=750
LR=1e-4
BATCH_SIZE=128

# Losses
LAMBDA_RESIDUAL=1.0
LAMBDA_ENDPOINT=0.0
LAMBDA_SPARAM=0.0
LAMBDA_PHASE=0.0
LAMBDA_PHASE_GRAD=0.0
SPARAM_MODE="modal"

# Spatial focus
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

echo "Starting FAST training job: ${VERSION}"
echo "Using preprocessed data from: ${DATA_ROOT}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NPROC_PER_NODE}"

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="$SLURM_JOB_ID" \
  /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --all-losses-from-start \
  --data-root "${DATA_ROOT}" \
  --use-fast-dataset \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --train-fraction 0.9 \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --train-num-workers 8 \
  --val-num-workers 4 \
  --prefetch-factor 4 \
  \
  --phaseA-epochs 0 \
  --phaseB-epochs 0 \
  \
  --lambda-phase "${LAMBDA_PHASE}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode "${SPARAM_MODE}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs 0 \
  --residual-start-phase "A" \
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
  --log-every 5 \
  \
  --no-include-sdf \
  --use-checkpoint \
  \
  \
  --tqdm \
  --tqdm-mininterval 15.0 \
  \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project PhotonFlow \
  \
  --no-config

echo "FAST training job completed."
