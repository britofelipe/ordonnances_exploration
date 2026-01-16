#!/bin/bash

#SBATCH --job-name=ordonnances
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"

# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

export PYTHONNOUSERSITE=1

# Load the conda environment
source activate ordonnances

# Display the installed packages
python -m pip list
which tesseract

# Run generation (5000 examples)
python generate_ordo_mimic.py --count 5000 --out output_treino