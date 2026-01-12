#!/usr/bin/env bash
#SBATCH --job-name=rayfield-train
#SBATCH --partition=a5500_w
#SBATCH --gres=gpu:nvidia_rtx_a5500:8
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rayfield_%j.out
#SBATCH --error=logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodelist=centurion[07-09]

source ~/.bashrc
conda activate PBFM
cd $SLURM_SUBMIT_DIR

# -----------------------
# Run config (curriculum)
# -----------------------
VERSION="euler_complexphys_v4_384_dx20"
HIDDEN_SIZE=16

# Dataset assumptions:
# - No PML in stored arrays
# - Data spatial size is 384x384
# - Grid spacing dx = 1/20 um/px = 0.05
EPOCHS=1500
LR=2e-4
BATCH_SIZE=48

DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"
SWEEPS="euler_bend_sweep_384"  # comma-separated; leave empty for all sweeps
USE_SHARDS="--use-shards"       # set to "--no-use-shards" to read raw folders
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"

# Curriculum boundaries
# Phase A: FM + endpoint only
# Phase B: add residual + sparams
# Phase C: add phase + phase-grad
PHASE_A_EPOCHS=100
PHASE_B_EPOCHS=250

# Loss weights (final magnitudes)
LAMBDA_ENDPOINT=1.0
LAMBDA_RESIDUAL=0.8
LAMBDA_PHASE=0.1
LAMBDA_PHASE_GRAD=0.05
LAMBDA_SPARAM=0.4

# Warmups / start phases
RESIDUAL_WARMUP=100
PHASE_WARMUP=200
ENDPOINT_WARMUP=10
PHASE_GRAD_WARMUP=200

# Phase-grad sparsity (speed knob)
PHASE_GRAD_EVERY=4
SPARAM_EVERY=1
RESIDUAL_START_PHASE="B"
PHASE_GRAD_START_PHASE="C"
PHASE_START_PHASE="C"
SPARAM_WARMUP=100
SPARAM_START_PHASE="B"

# Spatial focus knobs
# Note: raising W_MIN prevents “ignoring background” by giving non-device pixels real weight.
DEVICE_FOCUS_MAX=3
DEVICE_FOCUS_WARMUP=25
DEVICE_FOCUS_HOLD=200
DEVICE_FOCUS_DECAY=500

W_MIN=0.5
EPS_THR=2.5
DILATE=15

echo "Starting training job: ${VERSION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="$SLURM_JOB_ID" \
  /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  ${USE_SHARDS} \
  --include-sweeps "${SWEEPS}" \
  --shard-subdir "${SHARD_SUBDIR}" \
  --shard-index-name "${SHARD_INDEX_NAME}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --train-fraction 0.9 \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --amp \
  --no-use-checkpoint \
  --batch-size "${BATCH_SIZE}" \
  --train-num-workers 1 \
  --val-num-workers 1 \
  --prefetch-factor 4 \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "${RESIDUAL_START_PHASE}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --phase-start-phase "${PHASE_START_PHASE}" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --phase-grad-warmup-epochs "${PHASE_GRAD_WARMUP}" \
  --phase-grad-every "${PHASE_GRAD_EVERY}" \
  --phase-grad-start-phase "${PHASE_GRAD_START_PHASE}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
  --sparam-mode head \
  --sparam-warmup-epochs "${SPARAM_WARMUP}" \
  --sparam-every "${SPARAM_EVERY}" \
  --sparam-start-phase "${SPARAM_START_PHASE}" \
  --device-focus-max "${DEVICE_FOCUS_MAX}" \
  --device-focus-warmup-epochs "${DEVICE_FOCUS_WARMUP}" \
  --device-focus-hold-epochs "${DEVICE_FOCUS_HOLD}" \
  --device-focus-decay-epochs "${DEVICE_FOCUS_DECAY}" \
  --w-min "${W_MIN}" \
  --eps-thr "${EPS_THR}" \
  --dilate "${DILATE}" \
  --ckpt-every 100 \
  --eval-every 10 \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project PhotonFlow \
  --no-config \
  --val-batches 32 \
  --pml-cells 0 \
  --physics-features \
  --complex-unet \
  --dx 0.05 \
  --warmup-epochs 2 \
  --warmup-start-factor 0.1 \
  --min-lr 5e-6 \
  --include-sdf \
  --normalize-sdf \
  --sdf-thr-eps 3.0 \
  --sdf-feature clip_exp \
  --sdf-sigma-px 3 \
  --phase-amp-tau 0.4

echo "Training job completed."
