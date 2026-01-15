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
conda create --name fhir_env python=3.9 --force

# Activate the environment
source activate fhir_env

# Install the required libraries from a requirement file, using pypi
# Instala Unsloth compatível com CUDA 12.1 e Torch 2.3.0
pip install --upgrade pip setuptools wheel
pip install "sympy>=1.12"
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets pdf2image
pip install "numpy<2.0"
pip install -r requirements.txt 