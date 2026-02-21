#!/usr/bin/env bash

#SBATCH --job-name=rayfield-finetune
#SBATCH --partition=a100_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=15
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
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
# Data lives in home directory (accessible from all nodes)
DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"

# -----------------------
# Run config: finetune from epoch 400 with lower LR + stronger residual
# -----------------------
VERSION="coupler_complexphys_v10_fwd_ft1"
RESUME_FROM="logs_physics_unet_pbfm/coupler_complexphys_v10_fwd/checkpoints/0000400.pt"

HIDDEN_SIZE=32

# Data - sharded NPZ files (directional_coupler_sweep)
USE_SHARDS="--use-shards"
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"
INCLUDE_SWEEPS="directional_coupler_sweep_v2"

# Train
EPOCHS=600
LR=1e-5
BATCH_SIZE=120
WEIGHT_DECAY=1e-4

# Curriculum: already in PhaseC from previous run, keep epochs low
PHASE_A_EPOCHS=0
PHASE_B_EPOCHS=0

# Losses: FM + Helmholtz residual (bumped from 1.0 → 1.5)
# No warmup needed — residual already at full weight from epoch 400.
LAMBDA_RESIDUAL=1.5
RESIDUAL_WARMUP=0
LAMBDA_ENDPOINT=0.0
LAMBDA_SPARAM=0.0
LAMBDA_PHASE=0.0
LAMBDA_PHASE_GRAD=0.0
SPARAM_MODE="project"

# Spatial focus knobs
# Relaxed from v3: coupler coupling gap is cladding material but critically
# important. Lower focus lets the model learn evanescent/cladding fields better.
DEVICE_FOCUS_MAX=2.0
DEVICE_FOCUS_WARMUP=25
DEVICE_FOCUS_HOLD=100
DEVICE_FOCUS_DECAY=200
W_MIN=0.5
EPS_THR=2.5
DILATE=15

# System/env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPUS=2

echo "Starting training job: ${VERSION}"
echo "Resuming from: ${RESUME_FROM}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NGPUS} WorldGPU=$((SLURM_NNODES * NGPUS))"

# Multi-node DDP setup
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 500))
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Quick GPU sanity check
echo ""
echo "=== GPU preflight ==="
nvidia-smi -L 2>&1
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1
python -c "
import os, torch
print('torch', torch.__version__)
print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('cuda.is_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    x = torch.tensor([1.0], device=f'cuda:{i}')
    torch.cuda.synchronize()
    print('ok', i, torch.cuda.get_device_name(i), 'x', x.item())
"
echo "=== end GPU preflight ==="
echo ""

srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${NGPUS} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  ${USE_SHARDS} \
  --shard-subdir "${SHARD_SUBDIR}" \
  --shard-index-name "${SHARD_INDEX_NAME}" \
  --include-sweeps "${INCLUDE_SWEEPS}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --train-fraction 0.90 \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --train-num-workers 6 \
  --val-num-workers 3 \
  --prefetch-factor 8 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "B" \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode "${SPARAM_MODE}" \
  \
  --device-focus-max "${DEVICE_FOCUS_MAX}" \
  --device-focus-warmup-epochs "${DEVICE_FOCUS_WARMUP}" \
  --device-focus-hold-epochs "${DEVICE_FOCUS_HOLD}" \
  --device-focus-decay-epochs "${DEVICE_FOCUS_DECAY}" \
  --w-min "${W_MIN}" \
  --eps-thr "${EPS_THR}" \
  --dilate "${DILATE}" \
  \
  --ckpt-every 25 \
  --eval-every 25 \
  --val-batches 16 \
  --save-eval-samples \
  --save-eval-samples-limit 4 \
  \
  --pml-cells 0 \
  --physics-features \
  --complex-unet \
  --dx 0.05 \
  --unroll-steps 0 \
  --fm_steps 100 \
  \
  --warmup-epochs 0 \
  --min-lr 1e-6 \
  --weight-decay "${WEIGHT_DECAY}" \
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
  --no-config \
  \
  --eps-core 5.8 \
  --eps-clad 2.09 \
  \
  --resume-from "${RESUME_FROM}"

echo "Training job completed."
