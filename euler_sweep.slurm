#!/usr/bin/env bash
#SBATCH --job-name=euler_sweep
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/euler_sweep_%j.out
#SBATCH --error=logs/euler_sweep_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

# Environment
source ~/.bashrc
conda activate PBFM

# Prevent oversubscription inside each worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- run params
N_GEO=5000
N_PROCS=26            # Use most of the 32 allocated CPUs
RES=20                # matches script default
DPML="0.6666666667"
CROP_PX=384
# Write to a separate folder so we don't overwrite existing datasets.
OUT_DIR="/dartfs-hpc/rc/home/j/f0071mj/rayfield/Data/euler_bend_sweep_${CROP_PX}"

# Optional: include net-turn devices (adjacent-edge I/O) in the sweep.
# Set INCLUDE_TURNS=1 to enable.
INCLUDE_TURNS=1
TURN_PROB=0.7
END_TURN_DEG=90.0

# ---- run
TURN_FLAGS=""
if [ "${INCLUDE_TURNS}" -eq 1 ]; then
  TURN_FLAGS="--include-turns --turn-prob ${TURN_PROB} --end-turn-deg ${END_TURN_DEG}"
fi

python -u FDTD/euler_bend_sweep.py \
  --out-dir ${OUT_DIR} \
  --n-geo ${N_GEO} \
  --n-procs ${N_PROCS} \
  --resolution ${RES} \
  --dpml ${DPML} \
  --crop-px ${CROP_PX} \
  ${TURN_FLAGS} \
  --compress \
  --shard-size 100 \
  --queue-max 128 \
  --index-every-shard

