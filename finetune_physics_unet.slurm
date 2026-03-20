#!/usr/bin/env bash

#SBATCH --job-name=rayfield-v6
#SBATCH --partition=h200_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/dartfs-hpc/rc/home/j/f0071mj/rayfield/logs/rayfield_%j.out
#SBATCH --error=/dartfs-hpc/rc/home/j/f0071mj/rayfield/logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate PBFM
RUN_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield"
CHECKPOINT_ROOT="/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm"
cd "${RUN_ROOT}"
mkdir -p logs
mkdir -p "${CHECKPOINT_ROOT}"
export RAYFIELD_CKPT_ROOT="${CHECKPOINT_ROOT}"

# -----------------------
# Copy data to local scratch for faster I/O
# -----------------------
NFS_DATA="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"
LOCAL_DATA="/scratch/${SLURM_JOB_ID}/rayfield_data"
echo "Copying data to local scratch: ${LOCAL_DATA}"
mkdir -p "${LOCAL_DATA}"
rsync -a "${NFS_DATA}/" "${LOCAL_DATA}/"
echo "Data copy complete ($(du -sh "${LOCAL_DATA}" | cut -f1))"
DATA_ROOT="${LOCAL_DATA}"

# -----------------------
# Run config: v6 — final dataset (res=20, 9 devices, 5 wavelengths, all rect 480x160)
#
# Key changes vs v5:
#   - New dataset: unified_sweep_final (81K samples, 9 device types, 5 wavelengths)
#   - Resolution: res=20 (dx=0.05), tensors 480x160
#   - λ_phase 0.1→0.0 (saturated at 0.7° in v5, complex UNet handles phase natively)
#   - λ_sparam 0.1→0.5 (was underpowered — S-params had 30-50% errors at v5 end)
#   - S-param mode: head (learned, not projection)
#   - Curriculum compressed for 80 epochs: PhaseA=5, PhaseB=15, PhaseC=15+
#   - S-param warmup 30→5 to match the shorter training schedule
#   - No resume — fresh training from scratch on new dataset
#   - Hidden 40→48 (~90M params, 1.44x capacity for 2.7x more data)
#   - Epochs reduced to 80 for faster turnaround on the 50k-sample training set
# -----------------------
VERSION="PHASE_V1_train1"

# Architecture
HIDDEN_SIZE=40
NUM_RES_BLOCKS=2
CHANNEL_MULT="1,2,4,8"
NUM_HEADS=8
ATTN_RESOLUTIONS="8"
MODEL_DROPOUT=0.0

# Data — final dataset (all rect, res=20)
USE_SHARDS="--use-shards"
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"
INCLUDE_SWEEPS="unified_sweep_final"
EXCLUDE_DEVICES=""
INCLUDE_WAVELENGTHS=""

# Train
EPOCHS=150
LR=1e-4
BATCH_SIZE=160
WEIGHT_DECAY=5e-5

# Curriculum — compressed for 80 epochs on the larger final dataset
PHASE_A_EPOCHS=15     # FM only
PHASE_B_EPOCHS=40    # FM+Residual (epochs 5-15), Phase C starts at epoch 15

# Time-gating: physics losses only when t >= 0.5
T_PHYSICS_MIN=0.5

# Logit-normal time sampling
T_SAMPLE_MODE="mixture"
T_SAMPLE_LOC=0.0
T_SAMPLE_SCALE=1.0

# Residual t-power weighting
RESIDUAL_T_POWER=4.0

# ODE sampler time grid
TIME_GRID="linear"

# Losses — FM + Residual + S-param (phase dropped, endpoint disabled)
LAMBDA_RESIDUAL=1.0
RESIDUAL_WARMUP=5

# Endpoint loss DISABLED
LAMBDA_ENDPOINT=0.0
ENDPOINT_START_PHASE="C"
ENDPOINT_WARMUP=0

# Phase loss DISABLED — complex UNet handles phase natively, saturated at 0.7° in v5
LAMBDA_PHASE=0.0
LAMBDA_PHASE_GRAD=0.0
PHASE_WARMUP=0
PHASE_GRAD_WARMUP=0

# S-param: learned head mode, 5x weight increase vs v5
LAMBDA_SPARAM=0.4       # v5: 0.1 — S-params were underpowered (30-50% |S| errors)
SPARAM_MODE="head"
SPARAM_WARMUP=5        # shorter warmup for the faster 80-epoch schedule
SPARAM_START_PHASE="C"

# Spatial focus knobs (disabled — let model learn full cladding fields)
DEVICE_FOCUS_MAX=0.0
DEVICE_FOCUS_WARMUP=25
DEVICE_FOCUS_HOLD=100
DEVICE_FOCUS_DECAY=200
W_MIN=0.5
EPS_THR=2.5
DILATE=15
INTERFACE_THR=1.0

# System/env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPUS=2

echo "Starting training job: ${VERSION}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NGPUS} WorldGPU=$((SLURM_NNODES * NGPUS))"

# Multi-node DDP setup
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 500))
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Quick GPU sanity check (srun so it runs on an allocated node)
echo ""
echo "=== GPU preflight ==="
srun --ntasks=1 --nodes=1 bash -c 'nvidia-smi -L 2>&1; python -c "
import os, torch
print(\"torch\", torch.__version__)
print(\"CUDA_VISIBLE_DEVICES\", os.environ.get(\"CUDA_VISIBLE_DEVICES\"))
print(\"cuda.is_available\", torch.cuda.is_available())
print(\"device_count\", torch.cuda.device_count())
"'
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
  --exclude-devices "${EXCLUDE_DEVICES}" \
  --include-wavelengths "${INCLUDE_WAVELENGTHS}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-res-blocks "${NUM_RES_BLOCKS}" \
  --channel-mult "${CHANNEL_MULT}" \
  --num-heads "${NUM_HEADS}" \
  --attn-resolutions "${ATTN_RESOLUTIONS}" \
  --model-dropout "${MODEL_DROPOUT}" \
  --train-fraction 0.90 \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --amp-dtype bfloat16 \
  --train-num-workers 10 \
  --val-num-workers 4 \
  --prefetch-factor 4 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "B" \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-start-phase "${ENDPOINT_START_PHASE}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --phase-start-phase "C" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --phase-grad-warmup-epochs "${PHASE_GRAD_WARMUP}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode "${SPARAM_MODE}" \
  --sparam-warmup-epochs "${SPARAM_WARMUP}" \
  --sparam-start-phase "${SPARAM_START_PHASE}" \
  \
  --t-physics-min "${T_PHYSICS_MIN}" \
  --t-sample-mode "${T_SAMPLE_MODE}" \
  --t-sample-loc "${T_SAMPLE_LOC}" \
  --t-sample-scale "${T_SAMPLE_SCALE}" \
  --residual-t-power "${RESIDUAL_T_POWER}" \
  --time-grid "${TIME_GRID}" \
  \
  --device-focus-max "${DEVICE_FOCUS_MAX}" \
  --device-focus-warmup-epochs "${DEVICE_FOCUS_WARMUP}" \
  --device-focus-hold-epochs "${DEVICE_FOCUS_HOLD}" \
  --device-focus-decay-epochs "${DEVICE_FOCUS_DECAY}" \
  --w-min "${W_MIN}" \
  --eps-thr "${EPS_THR}" \
  --dilate "${DILATE}" \
  --interface-thr "${INTERFACE_THR}" \
  \
  --ckpt-every 20 \
  --eval-every 5 \
  --val-batches 4 \
  --save-eval-samples \
  --save-eval-samples-limit 2 \
  --sample-eval-limit 2 \
  \
  --pml-cells 0 \
  --physics-features \
  --complex-unet \
  --use-checkpoint \
  --dx 0.05 \
  --unroll-steps 0 \
  --fm_steps 50 \
  \
  --warmup-epochs 5 \
  --min-lr 5e-6 \
  --weight-decay "${WEIGHT_DECAY}" \
  --log-every 1 \
  \
  --no-include-sdf \
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
  --normalize-eps \
  \
  --eps-core 5.8 \
  --eps-clad 2.09 \

echo "Training job completed."
