#!/usr/bin/env bash
# =============================================================================
# REAL-VALUED UNet arm of the real-vs-complex paper comparison.
#
# Loss: FM + phase  (no residual, no sparam, no endpoint, no physics features)
# Arch: Real PhysicsUNet, HIDDEN_SIZE=56, channel_mult=1,2,4,8, attn ds=4,8
#       -> 63.3 M params  (param-matched against complex arm at h=40 -> 64.6 M)
#
# Pair with: train_complex_unet_phase.slurm  (HIDDEN_SIZE=40, 64.6 M params)
# Identical dataset / epochs / eval / seeds / loss for a clean A/B.
#
# Param-matching rationale:
#   - ComplexConv2d == 2 real convs internally, so for the same channel count
#     it has ~2x the params of a real conv.
#   - Channel-matched comparison favors complex unfairly. Param-matched is the
#     defensible control: if complex still wins, the win is from architecture
#     not capacity.
# =============================================================================

#SBATCH --job-name=rayfield-real
#SBATCH --partition=v100_preemptable
#SBATCH --nodelist=p[02-04]
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
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

VERSION="REAL_VS_COMPLEX_real_h56_phase_v100x12"

# -----------------------
# Architecture (REAL arm) — param-matched to complex arm (63.3 M vs 64.6 M)
# -----------------------
HIDDEN_SIZE=56
NUM_RES_BLOCKS=3
CHANNEL_MULT="1,2,4,8"
NUM_HEADS=8
ATTN_RESOLUTIONS="4,8"
MODEL_DROPOUT=0.0

USE_SHARDS="--use-shards"
USE_INDEX_SPLIT="--use-index-split"
SHARD_SUBDIR="shards"
SHARD_INDEX_NAME="index.json"
INCLUDE_SWEEPS="unified_sweep_mmi_ybranch_dc_7500_each_1p55um"
EXCLUDE_DEVICES="straight,taper,sbend,euler_bend,circular_bend,crossing"
INCLUDE_WAVELENGTHS="1.55"
TRAIN_FRACTION=0.9

# Per-GPU batch=4 -> global batch = 4 * 12 GPUs = 48 (V100 32GB, safe with h=56)
EPOCHS=300
LR=1e-4
BATCH_SIZE=48
WEIGHT_DECAY=2e-5

PHASE_A_EPOCHS=10
PHASE_B_EPOCHS=20

T_PHYSICS_MIN=0.0
T_SAMPLE_MODE="mixture"
T_SAMPLE_LOC=0.8
T_SAMPLE_SCALE=0.8
RESIDUAL_T_POWER=0
TIME_GRID="linear"

# Losses — FM + PHASE only (matches complex arm exactly)
LAMBDA_RESIDUAL=0.0
RESIDUAL_WARMUP=0

LAMBDA_ENDPOINT=0.0
ENDPOINT_START_PHASE="C"
ENDPOINT_WARMUP=0

LAMBDA_PHASE=0.1
LAMBDA_PHASE_GRAD=0.0
PHASE_WARMUP=15
PHASE_GRAD_WARMUP=0

LAMBDA_SPARAM=0.0
SPARAM_MODE="head"
SPARAM_WARMUP=0
SPARAM_START_PHASE="C"

DEVICE_FOCUS_MAX=1.2
DEVICE_FOCUS_WARMUP=20
DEVICE_FOCUS_HOLD=20
DEVICE_FOCUS_DECAY=60
W_MIN=0.8
EPS_THR=2.5
DILATE=15
INTERFACE_THR=1.0

EVAL_SAMPLE_MODE="fixed"
EVAL_SAMPLE_INDEX_SEED=0
EVAL_FLOW_NOISE_SEED=0
EVAL_SAMPLES_PER_DEVICE=8
SAMPLE_EVAL_LIMIT=24
PHYSICAL_RATIO_TARGET=2
CKPT_BEST_METRIC="worst_ratio_p95"
CKPT_EVERY=10

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPUS=4
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${VERSION}/checkpoints"
RESUME_ARGS=()
mkdir -p "${CHECKPOINT_DIR}"
echo "Starting fresh (no resume checkpoint)."

echo "Starting training job: ${VERSION}"
echo "Nodes=${SLURM_NNODES} GPUs/node=${NGPUS} WorldGPU=$((SLURM_NNODES * NGPUS))"

MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 500))
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

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
  ${USE_INDEX_SPLIT} \
  --version "${VERSION}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-res-blocks "${NUM_RES_BLOCKS}" \
  --channel-mult "${CHANNEL_MULT}" \
  --num-heads "${NUM_HEADS}" \
  --attn-resolutions "${ATTN_RESOLUTIONS}" \
  --model-dropout "${MODEL_DROPOUT}" \
  --train-fraction "${TRAIN_FRACTION}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --amp \
  --amp-dtype float16 \
  --train-num-workers 4 \
  --val-num-workers 4 \
  --prefetch-factor 4 \
  \
  --phaseA-epochs "${PHASE_A_EPOCHS}" \
  --phaseB-epochs "${PHASE_B_EPOCHS}" \
  \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --residual-warmup-epochs "${RESIDUAL_WARMUP}" \
  --residual-start-phase "A" \
  --lambda-endpoint "${LAMBDA_ENDPOINT}" \
  --endpoint-start-phase "${ENDPOINT_START_PHASE}" \
  --endpoint-warmup-epochs "${ENDPOINT_WARMUP}" \
  --lambda-phase "${LAMBDA_PHASE}" \
  --phase-warmup-epochs "${PHASE_WARMUP}" \
  --phase-start-phase "A" \
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
  --ckpt-every "${CKPT_EVERY}" \
  --eval-every 5 \
  --val-batches 4 \
  --save-eval-samples \
  --save-eval-samples-limit 2 \
  --sample-eval-limit "${SAMPLE_EVAL_LIMIT}" \
  --eval-sample-mode "${EVAL_SAMPLE_MODE}" \
  --eval-sample-index-seed "${EVAL_SAMPLE_INDEX_SEED}" \
  --eval-flow-noise-seed "${EVAL_FLOW_NOISE_SEED}" \
  --eval-samples-per-device "${EVAL_SAMPLES_PER_DEVICE}" \
  --physical-ratio-target "${PHYSICAL_RATIO_TARGET}" \
  --ckpt-best-metric "${CKPT_BEST_METRIC}" \
  \
  --pml-cells 0 \
  --no-physics-features \
  --no-complex-unet \
  --use-checkpoint \
  --dx 0.05 \
  --unroll-steps 0 \
  --fm_steps 100 \
  \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip 5.0 \
  --log-every 1 \
  \
  --include-sdf \
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
  "${RESUME_ARGS[@]}" \
  \
 \

echo "Training job completed."
