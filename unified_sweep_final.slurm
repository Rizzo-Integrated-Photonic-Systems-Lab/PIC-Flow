#!/usr/bin/env bash
#SBATCH --job-name=unified_sweep_final
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/unified_sweep_final_%j.out
#SBATCH --error=logs/unified_sweep_final_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Final Production Dataset: 9 Device Types, ALL RECTANGULAR, res=20, 4 Wavelengths
# =============================================================================
#
# Single domain (480x160 px = 24x8 µm @ res=20):
#   - Straight waveguides   (200 geom)  [2D params]
#   - Tapers                (1000 geom) [3D params]
#   - S-bends               (1000 geom) [3D params]
#   - Y-branches            (2000 geom) [5D params]
#   - Directional couplers  (3000 geom) [5D params]
#   - 2x2 MMIs              (4000 geom) [5D params]
#   - Euler bends            (750 geom)  [2D params, R=2-3.5 µm]
#   - Circular bends         (750 geom)  [2D params, R=2-3.5 µm]
#   - Waveguide crossings    (1500 geom) [2D params]
#
# Total: 14,200 geometries x 4 wavelengths = 56,800 simulations
# All tensors: 480x160 px (no padding needed for training)
# Split: 80/10/10 by geometry_id
#
# Usage:
#   First run:  sbatch unified_sweep_final.slurm
#   Resubmit:   sbatch unified_sweep_final.slurm   (auto-resumes)
# =============================================================================

# Environment
source ~/.bashrc
conda activate PBFM

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- Run parameters ----

# Device counts (14,200 total geometries)
N_STRAIGHT=200
N_TAPER=1000
N_MMI=4000
N_SBEND=1000
N_YBRANCH=2000
N_DC=3000
N_EULER=750
N_CIRCULAR=750
N_CROSSING=1500

# Number of parallel workers (leave headroom for OS/IO)
N_PROCS=26

# Simulation resolution (pixels per um)
RESOLUTION=20

# Interior crop sizes — ALL rectangular
CROP_X_PX=480
CROP_Y_PX=160

# DFT decay tolerance
DECAY_TOL=1e-5

# Sharding
SHARD_SIZE=100
QUEUE_MAX=128
COMPRESS="--compress"

# Seeds (deterministic — same seeds across resumes)
SEED=42
SPLIT_SEED=123

# Output directory
OUT_DIR="Data/unified_sweep_final"

# ---- Create log directory ----
mkdir -p logs

# ---- Auto-detect resume ----
RESUME_FLAG=""
if [ -f "${OUT_DIR}/shards/index.json" ]; then
    EXISTING=$(python3 -c "import json; print(len(json.load(open('${OUT_DIR}/shards/index.json'))))" 2>/dev/null || echo "0")
    echo "Found existing index with ${EXISTING} samples — resuming"
    RESUME_FLAG="--resume"
else
    echo "No existing index — starting fresh"
fi

# ---- Run ----
echo "=========================================="
echo "Final Dataset Generation (all rect 480x160)"
echo "=========================================="

python -u FDTD/unified_sweep.py \
  --out-dir "${OUT_DIR}" \
  --n-straight ${N_STRAIGHT} \
  --n-taper ${N_TAPER} \
  --n-mmi ${N_MMI} \
  --n-sbend ${N_SBEND} \
  --n-ybranch ${N_YBRANCH} \
  --n-directional-coupler ${N_DC} \
  --n-euler-bend ${N_EULER} \
  --n-circular-bend ${N_CIRCULAR} \
  --n-crossing ${N_CROSSING} \
  --resolution ${RESOLUTION} \
  --crop-x-px ${CROP_X_PX} \
  --crop-y-px ${CROP_Y_PX} \
  --decay-tol ${DECAY_TOL} \
  --n-procs ${N_PROCS} \
  --shard-size ${SHARD_SIZE} \
  --queue-max ${QUEUE_MAX} \
  --seed ${SEED} \
  --split-seed ${SPLIT_SEED} \
  ${COMPRESS} \
  --index-every-shard \
  ${RESUME_FLAG}

echo "=========================================="
echo "Sweep run complete!"
echo "Output: ${OUT_DIR}/shards/"
echo "Index:  ${OUT_DIR}/shards/index.json"
echo "Geoms:  ${OUT_DIR}/geometries.jsonl"
echo "=========================================="

# Report progress
if [ -f "${OUT_DIR}/shards/index.json" ]; then
    TOTAL=$(python3 -c "import json; print(len(json.load(open('${OUT_DIR}/shards/index.json'))))" 2>/dev/null || echo "?")
    echo "Total samples so far: ${TOTAL} / 56800"
    if [ "${TOTAL}" != "56800" ] && [ "${TOTAL}" != "?" ]; then
        echo "*** Not yet complete — resubmit: sbatch unified_sweep_final.slurm ***"
    else
        echo "*** Dataset generation COMPLETE ***"
    fi
fi
