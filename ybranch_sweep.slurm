#!/usr/bin/env bash
#SBATCH --job-name=ybranch_sweep
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ybranch_sweep_%j.out
#SBATCH --error=logs/ybranch_sweep_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

# Environment
source ~/.bashrc
conda activate PBFM

# Prevent oversubscription inside each worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- run params (tune these)
N_GEO=5000
N_PROCS=26            # <= cpus-per-task; leave some headroom for OS/IO
RES=32
SHARD_SIZE=100        # samples per shard (each geometry -> 1 sample)
QUEUE_MAX=128          # bounds temp file footprint
COMPRESS="--compress" # comment out if you want faster, bigger shards

# ---- run
python -u FDTD/ybranch_sweep.py \
  --n-geo ${N_GEO} \
  --n-procs ${N_PROCS} \
  --resolution ${RES} \
  --shard-size ${SHARD_SIZE} \
  --queue-max ${QUEUE_MAX} \
  ${COMPRESS} \
  --index-every-shard



