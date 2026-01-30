#!/bin/bash
#SBATCH --job-name=neff_extract_local
#SBATCH --partition=preemptable
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/neff_extract_local_%j.out
#SBATCH --error=logs/neff_extract_local_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=f0071mj@dartmouth.edu


# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job started at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: $SLURM_MEM_PER_NODE"

# Environment setup (like train_physics_unet.slurm)
source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

# Verify environment
echo "Python version: $(python --version)"
echo "Tidy3D available: $(python -c 'import tidy3d; print("Yes")' 2>/dev/null || echo "No")"

# Change to working directory
cd /dartfs-hpc/rc/home/j/f0071mj/rayfield

# Run the neff extraction script (local solver)
echo "Starting neff_extract.py with local solver at $(date)..."
echo "This may take several hours with 3,423 computations..."
python FDTD/neff_extract.py --num-processes 30

# Print completion information
echo "Job completed at $(date)"
echo "Check neff_tables/ directory for output files"
