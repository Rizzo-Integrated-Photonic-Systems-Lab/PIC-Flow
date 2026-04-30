#!/usr/bin/env bash

# Finetune from PHASE_V1_train3 with increased residual weight to push residual ratio toward 1x

#SBATCH --job-name=rayfield-resfinetune
#SBATCH --partition=a5500
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=centurion[02-05]
#SBATCH --gres=gpu:nvidia_rtx_a5500:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
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

DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"

# -----------------------
# Finetune: PHASE_V1_train3 (ep500) → higher λ_residual to improve residual ratio
# - Resume from 0000500.pt
# - LAMBDA_RESIDUAL 2→4
# - Lower LR, fewer epochs, fresh optimizer
# -----------------------
VERSION="PHASE_V1_train3_residual_finetune"
RESUME_SRC="${CHECKPOINT_ROOT}/PHASE_V1_train3/checkpoints/0000500.pt"
RESUME_DIR="${RUN_ROOT}/checkpoints_resume"
mkdir -p "${RESUME_DIR}"
RESUME_FROM="${RESUME_DIR}/0000500.pt"
# Copy to project dir so all compute nodes can read (lab path has permission issues on centurion)
if [ ! -f "${RESUME_FROM}" ]; then
  echo "Copying checkpoint to ${RESUME_FROM}..."
  cp "${RESUME_SRC}" "${RESUME_FROM}" || { echo "Copy failed. Run manually: cp ${RESUME_SRC} ${RESUME_FROM}"; exit 1; }
fi

# Architecture (must match base run)
HIDDEN_SIZE=32
NUM_RES_BLOCKS=3
CHANNEL_MULT="1,2,4,8"
NUM_HEADS=4
ATTN_RESOLUTIONS="8"
MODEL_DROPOUT=0.0

# Data
USE_SHARDS="--use-shards"
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"
INCLUDE_SWEEPS="unified_sweep_final"
EXCLUDE_DEVICES="euler_bend,circular_bend,crossing"
INCLUDE_WAVELENGTHS=""

# Finetune: 150 ADDITIONAL epochs (resume from 500 → run to 650)
EPOCHS=400
LR=2e-5
BATCH_SIZE=160
WEIGHT_DECAY=5e-5

# Curriculum (already past Phase C from base run)
PHASE_A_EPOCHS=0
PHASE_B_EPOCHS=0
T_PHYSICS_MIN=0.5
T_SAMPLE_MODE="mixture"
T_SAMPLE_LOC=0.0
T_SAMPLE_SCALE=1.0
RESIDUAL_T_POWER=3
TIME_GRID="linear"

# Losses — INCREASED residual weight (main change)
LAMBDA_RESIDUAL=2.5
RESIDUAL_WARMUP=0
LAMBDA_ENDPOINT=0.0
ENDPOINT_START_PHASE="C"
ENDPOINT_WARMUP=0
LAMBDA_PHASE=0.0
LAMBDA_PHASE_GRAD=0.0
PHASE_WARMUP=0
PHASE_GRAD_WARMUP=0
LAMBDA_SPARAM=0.0
SPARAM_MODE="head"
SPARAM_WARMUP=0
SPARAM_START_PHASE="C"

# Spatial focus
DEVICE_FOCUS_MAX=0.0
DEVICE_FOCUS_WARMUP=25
DEVICE_FOCUS_HOLD=100
DEVICE_FOCUS_DECAY=200
W_MIN=0.5
EPS_THR=2.5
DILATE=15
INTERFACE_THR=1.0

# System
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPUS=8

echo "Finetuning: resume from ${RESUME_FROM}, λ_residual=${LAMBDA_RESIDUAL}"
echo "Version: ${VERSION}, Epochs=${EPOCHS}, LR=${LR}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NGPUS}"

MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 500))
export MASTER_ADDR MASTER_PORT

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
  --resume-from "${RESUME_FROM}" \
  --reset-optimizer \
  --reset-lr-on-resume \
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
  --amp-dtype float16 \
  --train-num-workers 2 \
  --val-num-workers 4 \
  --prefetch-factor 4 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "B" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --phase-start-phase "C" \
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
  --ckpt-every 25 \
  --eval-every 10 \
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
  --fm_steps 50 \
  \
  --warmup-epochs 3 \
  --min-lr 1e-6 \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip 5.0 \
  --log-every 1 \
  \
  --no-include-sdf \
  --tqdm \
  --tqdm-mininterval 15.0 \
  \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project PhotonFlow \
  \
  --normalize-eps \

echo "Finetuning job completed."
