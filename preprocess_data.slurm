#!/usr/bin/env bash
#SBATCH --job-name=preprocess
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield

source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

echo "Starting dataset preprocessing..."
echo "Using ${SLURM_CPUS_PER_TASK} CPUs"

python /dartfs-hpc/rc/home/j/f0071mj/rayfield/Model/preprocess_dataset.py \
    --data-root "/dartfs/rc/lab/R/RizzoA/f0071mj/Data" \
    --output-dir "/dartfs/rc/lab/R/RizzoA/f0071mj/Data_preprocessed/unified_sweep" \
    --include-sweeps "unified_sweep" \
    --shard-subdir "shards" \
    --num-workers 32 \
    --pml-cells 0 \
    --stats-samples 500

echo "Preprocessing complete!"
