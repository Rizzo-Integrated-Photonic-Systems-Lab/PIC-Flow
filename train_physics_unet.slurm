#!/usr/bin/env bash
# SLURM script to run Model/train.py (single GPU). The code is not DDP-ready,
# so we request one GPU and run plain python (no torchrun).

#SBATCH --job-name=physics_unet_v5
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rayfield_%j.out
#SBATCH --error=logs/rayfield_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate your env
source ~/.bashrc
conda activate PBFM

cd $SLURM_SUBMIT_DIR

# Hyperparameters
VERSION="physics_unet_v5"
HIDDEN_SIZE=32
EPOCHS=500
LR=5e-5
LAMBDA_RESIDUAL=0.6
RESIDUAL_WARMUP=25
BATCH_SIZE=16
DATA_ROOT="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data"

echo "Starting training job: ${VERSION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun python /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/train.py \
  --data-root "${DATA_ROOT}" \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --batch-size "${BATCH_SIZE}" \
  --ckpt-every 100 \
  --use-wandb \
  --wandb-entity jjquaratiello-dartmouth \
  --wandb-project Rayfield

echo "Training job completed."

