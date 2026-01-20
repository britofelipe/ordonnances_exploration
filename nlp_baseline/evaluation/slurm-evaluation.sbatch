#!/bin/bash

#SBATCH --job-name=bert-ev
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"


# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Create the environment using python 3.9
#conda create --name bert_env python=3.9 --force

# Activate the environment
source activate ordonnances
#pip install 'accelerate>=0.26.0'
#pip install datasets
#pip install transformers
#pip install 'accelerate>=0.26.0'
#pip uninstall -y torch torchvision torchaudio

# Display the installed packages using the conda install python interpreter
python -m pip list

#tesseract --list-langs
python bert_evaluation_struct.py