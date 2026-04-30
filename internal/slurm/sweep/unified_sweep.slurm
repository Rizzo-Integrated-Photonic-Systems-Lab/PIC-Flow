#!/usr/bin/env bash
#SBATCH --job-name=unified_sweep
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/unified_sweep_%j.out
#SBATCH --error=logs/unified_sweep_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

# =============================================================================
# Unified Multi-Device Photonic Dataset Generation
# =============================================================================
#
# Generates training data for 9 device types across two domain shapes:
#
#   Rectangular (480x160 px = 24x8 µm):
#     - Straight waveguides (200)
#     - Tapers (800)
#     - 2x2 MMIs (2500)
#     - S-bends (800)
#     - Y-branches (1500)
#     - Directional couplers (2500)
#
#   Square (320x320 px = 16x16 µm):
#     - Euler bends (1500)
#     - Circular bends (1500)
#     - Waveguide crossings (3000)
#
# Total: ~14,300 unique geometries x 5 wavelengths = ~71,500 simulations
#        D2/D4 augmentation applied during training (not at generation time)
#
# Split: 80/10/10 by geometry_id
#
# Per sample: eps, Ez (complex), src_mask, port_masks, S-parameters
# =============================================================================

# Environment
source ~/.bashrc
conda activate PBFM

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- Run parameters (tune these) ----

# Total unique geometries (scaled proportionally across 9 device types)
N_GEO=14300

# Number of parallel workers (leave headroom for OS/IO)
N_PROCS=26

# Simulation resolution (pixels per um) — meets Meep minimum of 18 for Si at 1550nm
RESOLUTION=20

# Interior crop size in pixels — rectangular domain (propagation x transverse)
CROP_X_PX=480
CROP_Y_PX=160

# Interior crop size in pixels — square domain (for bends/crossings)
CROP_SQ_PX=320

# DFT decay tolerance
DECAY_TOL=1e-5

# Sharding
SHARD_SIZE=100      # samples per shard
QUEUE_MAX=128       # bounds temp file footprint
COMPRESS="--compress"  # comment out for faster, bigger shards

# Seeds for reproducibility
SEED=42
SPLIT_SEED=123

# Output directory
OUT_DIR="Data/unified_sweep"

# ---- Create log directory ----
mkdir -p logs

# ---- Run ----
python -u FDTD/unified_sweep.py \
  --out-dir "${OUT_DIR}" \
  --n-geo ${N_GEO} \
  --resolution ${RESOLUTION} \
  --crop-x-px ${CROP_X_PX} \
  --crop-y-px ${CROP_Y_PX} \
  --crop-sq-px ${CROP_SQ_PX} \
  --decay-tol ${DECAY_TOL} \
  --n-procs ${N_PROCS} \
  --shard-size ${SHARD_SIZE} \
  --queue-max ${QUEUE_MAX} \
  --seed ${SEED} \
  --split-seed ${SPLIT_SEED} \
  ${COMPRESS} \
  --index-every-shard

echo "=========================================="
echo "Unified sweep complete!"
echo "Output: ${OUT_DIR}/shards/"
echo "Index:  ${OUT_DIR}/shards/index.json"
echo "Geoms:  ${OUT_DIR}/geometries.jsonl"
echo "=========================================="
