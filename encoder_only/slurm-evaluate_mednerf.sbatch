#!/bin/bash

#SBATCH --job-name=mednerf_eval
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

# Run MedNERF evaluation
# --model_path: use best_model (saved by trainer.save_model) or a specific checkpoint
python evaluate_on_mednerf.py \
    --model_path camembertBIO-ner-output/best_model \
    --output_dir mednerf_results \
    --batch_size 32 \
    --max_length 128 \
    --show_errors
