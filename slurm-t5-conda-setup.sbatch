#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00

# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Create the environment using python 3.9
conda create --name flan_t5_env python=3.9 --force

# Activate the environment
source activate flan_t5_env

# Install the required libraries from a requirement file, using pypi
pip install -r requirements.txt 
# You can also trigger "conda install ..." commands