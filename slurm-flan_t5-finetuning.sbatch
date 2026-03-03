#!/bin/bash

#SBATCH --job-name=flan_t5
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"

# Ensure module is visible 
source /etc/profile

# Load Anaconda
module load anaconda3/2022.10/gcc-13.1.0

# Prevent system site-packages from interfering
export PYTHONNOUSERSITE=1

# Activate your pre-created environment
source activate flan_t5_env

# Display the installed packages
python -m pip list

# Launch the fine-tuning script
python train_flan_t5_small.py