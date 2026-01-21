#!/bin/bash
#SBATCH --job-name=vis-gen
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=02:00:00
#SBATCH --output=visual/generation/logs/gen-%j.out
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "Starting Visual Data Generation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Load environment
source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
source activate ordonnances

# Check python version
python --version

echo "Environment loaded. Starting generation script..."
echo "Target Count: 1000 images"
echo "Output Dir: visual/dataset_v1"

# Run Generation
# We generate 1000 images for initial training of the detector
python visual/generation/generate_visual_data.py \
    --count 1000 \
    --out visual/dataset_v1 \
    --csv data/prescriptions_demo.csv \
    --blur 0.2 \
    --jpeg 0.3 \
    --stains 0.2 \
    --skew 0.0

echo ""
echo "=========================================="
echo "Job Finished at $(date)"
echo "=========================================="
