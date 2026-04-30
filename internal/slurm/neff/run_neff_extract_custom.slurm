#!/bin/bash
#SBATCH --job-name=neff_extract_local
#SBATCH --partition=preemptable
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/neff_extract_%j.out
#SBATCH --error=logs/neff_extract_%j.err
#SBATCH --chdir=/dartfs-hpc/rc/home/j/f0071mj/rayfield
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=f0071mj@dartmouth.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Default parameters (can be overridden with sbatch --export)
: ${NUM_PROCESSES:=30}
: ${WIDTH_MIN:=0.38}
: ${WIDTH_MAX:=2.00}
: ${LAMBDA_MIN:=1.45}
: ${LAMBDA_MAX:=1.65}

# Print job information
echo "Job started at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Parameters: processes=$NUM_PROCESSES, width=$WIDTH_MIN-$WIDTH_MAX, lambda=$LAMBDA_MIN-$LAMBDA_MAX"

# Environment setup (like train_physics_unet.slurm)
source ~/.bashrc
conda activate PBFM
cd "$SLURM_SUBMIT_DIR"

# Verify environment
echo "Python version: $(python --version)"
echo "Tidy3D available: $(python -c 'import tidy3d; print("Yes")' 2>/dev/null || echo "No")"

# Change to working directory
cd /dartfs-hpc/rc/home/j/f0071mj/rayfield

# Run the neff extraction script with parameters (local solver)
echo "Starting neff_extract.py with local solver at $(date)..."
echo "Parameters: processes=$NUM_PROCESSES, width=$WIDTH_MIN-$WIDTH_MAX, lambda=$LAMBDA_MIN-$LAMBDA_MAX"
python FDTD/neff_extract.py \
    --num-processes $NUM_PROCESSES \
    --width-min $WIDTH_MIN \
    --width-max $WIDTH_MAX \
    --lambda-min $LAMBDA_MIN \
    --lambda-max $LAMBDA_MAX

# Print completion information
echo "Job completed at $(date)"
echo "Check neff_tables/ directory for output files"
echo "Output files: neff_extract_${SLURM_JOB_ID}.out/err"
