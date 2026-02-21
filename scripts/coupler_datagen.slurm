#!/usr/bin/env bash
#SBATCH --job-name=dc_datagen
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/dc_datagen_%j.out
#SBATCH --error=logs/dc_datagen_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

# =============================================================================
# Directional Coupler Dataset Generation (rectangular grid)
# =============================================================================
#
# Generates DC-only training data with rectangular simulation windows:
#   - X (propagation): 512 px = 25.6 µm interior
#   - Y (transverse):  128 px = 6.4 µm interior
#
# Parameter ranges (5D geometry LHS):
#   wg_width:       0.375 - 0.600 µm (half-pixel quantized)
#   gap:            0.15 - 0.35 µm (stratified LHS, half-pixel quantized)
#   coupling_length: 5.0 - 12.0 µm (half-pixel quantized)
#   bend_length:    4.0 - 6.0 µm
#   lead_extra_gap: 0.8 - 2.5 µm
#
# Wavelengths swept separately per geometry (not part of LHS).
#
# Each (geometry, wavelength) sim yields 2 samples (port1 + port2 via symmetry).
# Total: N_GEO × N_WAVELENGTHS × 2 stored samples.
# =============================================================================

# Environment
source ~/.bashrc
conda activate PBFM

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- Run parameters ----
N_GEO=1250              # unique geometries (5D LHS)
N_WAVELENGTHS=10        # wavelengths per geometry (evenly spaced 1.40-1.60, ~22nm spacing)
                        # total FDTD sims: N_GEO * N_WAVELENGTHS = 12,500
                        # total samples:   N_GEO * N_WAVELENGTHS * 2 = 25,000
N_PROCS=26              # <= cpus-per-task; leave headroom for OS/IO
RES=20                  # pixels per µm

# Rectangular crop (non-PML interior)
CROP_X_PX=512           # propagation: 25.6 µm
CROP_Y_PX=128           # transverse:  6.4 µm

DPML="0.6666666667"     # 2/3 µm PML

SHARD_SIZE=100          # samples per shard
QUEUE_MAX=128           # bounds temp file footprint
COMPRESS="--compress"

OUT_DIR="/dartfs/rc/lab/R/RizzoA/f0071mj/Data/directional_coupler_sweep_v2"

# ---- Create directories ----
mkdir -p logs
mkdir -p "${OUT_DIR}"

# ---- Run ----
python -u FDTD/coupler_sweep.py \
  --out-dir "${OUT_DIR}" \
  --n-geo ${N_GEO} \
  --n-wavelengths ${N_WAVELENGTHS} \
  --n-procs ${N_PROCS} \
  --resolution ${RES} \
  --crop-x-px ${CROP_X_PX} \
  --crop-y-px ${CROP_Y_PX} \
  --dpml ${DPML} \
  --shard-size ${SHARD_SIZE} \
  --queue-max ${QUEUE_MAX} \
  ${COMPRESS} \
  --index-every-shard

echo "=========================================="
echo "Directional coupler sweep complete!"
echo "Output:  ${OUT_DIR}/shards/"
echo "Index:   ${OUT_DIR}/shards/index.json"
echo "Tip: review shard health before training"
echo "Grid:    ${CROP_X_PX}x${CROP_Y_PX} px (${RES} px/µm)"
echo "Samples: ~$((N_GEO * N_WAVELENGTHS * 2))"
echo "=========================================="
