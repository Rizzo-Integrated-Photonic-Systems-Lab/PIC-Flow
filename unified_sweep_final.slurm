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
# Final Production Dataset: selected device types, ALL RECTANGULAR, res=20, 1 Wavelength
# =============================================================================
#
# Single domain (480x160 px = 24x8 µm @ res=20):
#   - 2x2 MMIs              (7500 geom) [5D params]
#   - Y-branches            (7500 geom) [5D params]
#   - Directional couplers  (7500 geom) [5D params]
#
# Total: 22,500 geometries x 1 wavelength = 22,500 simulations
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

# Device counts (22,500 total geometries)
N_STRAIGHT=0
N_TAPER=0
N_MMI=7500
N_SBEND=0
N_YBRANCH=7500
N_DC=7500
N_EULER=0
N_CIRCULAR=0
N_CROSSING=0

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
OUT_DIR="Data/unified_sweep_mmi_ybranch_dc_7500_each_1p55um"

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

# Wavelength selection (unified_sweep.py uses a module-level constant)
WAVELENGTH_UM=1.55

# Build a temporary copy of unified_sweep.py with only the requested wavelength.
# Keep it under FDTD/ so relative module imports (e.g., straight_waveguide.*) still resolve.
TMP_SWEEP="FDTD/unified_sweep_1p55_tmp_${SLURM_JOB_ID}.py"
python3 - <<PY
from pathlib import Path
src = Path("FDTD/unified_sweep.py")
dst = Path("${TMP_SWEEP}")
text = src.read_text()
text = text.replace("WAVELENGTHS = [1.50, 1.525, 1.55, 1.575, 1.60]", "WAVELENGTHS = [${WAVELENGTH_UM}]")
dst.write_text(text)
print(f"Prepared temp sweep script: {dst}")
PY

cleanup() {
    rm -f "${TMP_SWEEP}"
}
trap cleanup EXIT

# ---- Run ----
echo "=========================================="
echo "Final Dataset Generation (all rect 480x160, lambda=${WAVELENGTH_UM}um)"
echo "=========================================="

python -u "${TMP_SWEEP}" \
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
    echo "Total samples so far: ${TOTAL} / 22500"
    if [ "${TOTAL}" != "22500" ] && [ "${TOTAL}" != "?" ]; then
        echo "*** Not yet complete — resubmit: sbatch unified_sweep_final.slurm ***"
    else
        echo "*** Dataset generation COMPLETE ***"
    fi
fi
