#!/bin/bash

#SBATCH --job-name=flan_t5
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
source activate bert_env

# Display installed packages
python -m pip list

# Run Flan-T5 fine-tuning script
python train_flan_t5_dsl_fhir.py \
    --data_dir output_mimic_fhir_ocr \
    --output_dir flan_t5_small_out \
    --model_name google/flan-t5-small