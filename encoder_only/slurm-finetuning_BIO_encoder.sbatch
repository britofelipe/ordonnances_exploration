#!/bin/bash

#SBATCH --job-name=camembertBIO
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"

# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Prevent the use of system site-packages
export PYTHONNOUSERSITE=1

# Activate the environment
source activate flan_t5_env

# Display installed packages
python -m pip list

# Run CamembertBIO fine-tuning script
python finetuning-encoder-only.py --data_path encoder_dataset/encoder_dataset.json --epochs 15 --lr 3e-5