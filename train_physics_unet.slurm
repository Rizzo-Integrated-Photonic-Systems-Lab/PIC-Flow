#!/usr/bin/env bash

#SBATCH --job-name=rayfield-train
#SBATCH --partition=v100_preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rayfield_%j.out
#SBATCH --error=logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate your env
source ~/.bashrc
conda activate PBFM

cd $SLURM_SUBMIT_DIR

# Hyperparameters
VERSION="no_endpoint_l40sx4"
HIDDEN_SIZE=32
EPOCHS=1000
LR=1.5e-4
LAMBDA_RESIDUAL=1.0
LAMBDA_PHASE=0.15
RESIDUAL_WARMUP=50
PHASE_WARMUP=50
BATCH_SIZE=32
DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"
DEVICE_FOCUS_MAX=3
DEVICE_FOCUS_WARMUP=100
DEVICE_FOCUS_HOLD=200
DEVICE_FOCUS_DECAY=300
W_MIN=0.2
EPS_THR=3.0
DILATE=15
LAMBDA_ENDPOINT=0.2
ENDPOINT_WARMUP=50
LAMBDA_PHASE_GRAD=0.1
PHASE_GRAD_WARMUP=100

echo "Starting training job: ${VERSION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun torchrun --standalone --nproc_per_node=4 /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --batch-size "${BATCH_SIZE}" \
  --device-focus-max "${DEVICE_FOCUS_MAX}" \
  --device-focus-warmup-epochs "${DEVICE_FOCUS_WARMUP}" \
  --device-focus-hold-epochs "${DEVICE_FOCUS_HOLD}" \
  --device-focus-decay-epochs "${DEVICE_FOCUS_DECAY}" \
  --w-min "${W_MIN}" \
  --eps-thr "${EPS_THR}" \
  --dilate "${DILATE}" \
  --ckpt-every 100 \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project Rayfield \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  --no-use-residual \
  --lambda-phase-grad "${LAMBDA_PHASE_GRAD}" \
  --phase-grad-warmup-epochs "${PHASE_GRAD_WARMUP}" 

echo "Training job completed."

