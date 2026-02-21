#!/usr/bin/env bash
# SLURM batch script to train PhysicsUNet
# Adjust partition/account/time/GPU as needed for your cluster.

#SBATCH --job-name=rayfield-train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

set -euo pipefail

# Optional: load modules; adjust for your cluster
# module load cuda/12.1

# Activate conda environment
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate PBFM
else
  echo "conda.sh not found; please load/activate env PBFM manually" >&2
  exit 1
fi

VERSION="physics_unet_v5"
HIDDEN_SIZE=16
EPOCHS=1000
LR=5e-5
LAMBDA_RESIDUAL=0.6
RESIDUAL_WARMUP=25
BATCH_SIZE=32
DATA_ROOT="/dartfs/rc/lab/R/RizzoA/f0071mj/Data"

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

