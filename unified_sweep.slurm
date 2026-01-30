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
# Generates training data for 8 device types:
#   - Straight waveguides (500)
#   - Tapers (750)
#   - Euler bends (500)
#   - Circular bends (500)
#   - S-bends (250)
#   - Y-branches (750) - supports splitter AND combiner modes
#   - Directional couplers (1000)
#   - Crossings (750)
#
# Total: 5,000 unique geometries × 4 wavelengths = 20,000 stored samples
#        D4 augmentation (8x) applied during training = 160,000 effective samples
#
# Split: 80/10/10 by geometry_id (4,000 train / 500 val / 500 test geometries)
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

# Total unique geometries (scaled proportionally across devices)
N_GEO=5000

# Number of parallel workers (leave headroom for OS/IO)
N_PROCS=26

# Simulation resolution (pixels per µm)
RESOLUTION=18

# PML thickness (µm)
DPML=1.0

# Simulation cell size (µm) - devices must fit in (cell - 2*dpml)
CELL_X=26.0
CELL_Y=26.0

# Non-PML crop size for quantized devices (ybranch, dir_coupler)
CROP_PX=432

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
  --dpml ${DPML} \
  --cell-x ${CELL_X} \
  --cell-y ${CELL_Y} \
  --crop-px ${CROP_PX} \
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
