#!/usr/bin/env bash
#SBATCH --job-name=rayfield-train
#SBATCH --partition=v100_preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rayfield_%j.out
#SBATCH --error=logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodelist=l40sx4

source ~/.bashrc
conda activate PBFM
cd $SLURM_SUBMIT_DIR

# -----------------------
# Run config (curriculum)
# -----------------------
VERSION="subset4k_hidden32_smoothlr_l40sx4"
HIDDEN_SIZE=32
EPOCHS=1500
LR=2.5e-4
BATCH_SIZE=32
DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"
SWEEPS="coupler_sweep,y_branch_sweep"  # comma-separated; leave empty for all sweeps
USE_SHARDS="--use-shards"       # set to "--no-use-shards" to read raw folders
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"

# Curriculum boundaries
PHASE_A_EPOCHS=50     # <=100: FM (+endpoint)
PHASE_B_EPOCHS=200     # 101..300: residual ramps; >300: phase ramps

# Loss weights (final magnitudes)
LAMBDA_ENDPOINT=1.0
LAMBDA_RESIDUAL=0.6
LAMBDA_PHASE=2
LAMBDA_PHASE_GRAD=2
LAMBDA_SPARAM=0.5

# Warmups (IMPORTANT: these warmups start AFTER the phase boundary)
RESIDUAL_WARMUP=50       # ramps from epoch 51 to 101 (if A=50)
PHASE_WARMUP=100         # ramps from epoch 201 to 301 (if B=200)
ENDPOINT_WARMUP=50
PHASE_GRAD_WARMUP=100    # ramps after phase C begins (only matters if lambda_phase_grad>0)
SPARAM_WARMUP=100        # ramps after phase C begins (only matters if lambda_sparam>0)

# Phase-grad sparsity (speed knob)
PHASE_GRAD_EVERY=1       # compute phase-grad every N train steps
SPARAM_EVERY=1           # compute sparam loss every N train steps (sparse speed knob)
SPARAM_START_PHASE="B"   # "B" = start after phase A, "C" = start after phase B (default)

# Spatial focus knobs
DEVICE_FOCUS_MAX=3
DEVICE_FOCUS_WARMUP=100
DEVICE_FOCUS_HOLD=200
DEVICE_FOCUS_DECAY=300
W_MIN=0.5
EPS_THR=3.0
DILATE=13

echo "Starting training job: ${VERSION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=4 \
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
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --amp \
  --batch-size "${BATCH_SIZE}" \
  --train-num-workers 8 \
  --val-num-workers 4 \
  --prefetch-factor 4 \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --phase-grad-warmup-epochs "${PHASE_GRAD_WARMUP}" \
  --phase-grad-every "${PHASE_GRAD_EVERY}" \
  --lambda-sparam "${LAMBDA_SPARAM}" \
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
  --eval-every 25 \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project Rayfield \
  --no-config \
  --val-batches 16 \
  --dx 0.03125 \
  --warmup-epochs 2 \
  --warmup-start-factor 0.1 \
  --min-lr 5e-6 

echo "Training job completed."
