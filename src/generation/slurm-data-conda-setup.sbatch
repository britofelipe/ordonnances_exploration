#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Deactivate any inherited environment (prevents "cannot remove current env" error)
source deactivate || conda deactivate || true

# Remove env if exists (to force clean install)
conda env remove -n ordonnances -y || true
# Create the environment using python 3.9
conda create --name ordonnances python=3.9 -y

# Activate the environment
source activate ordonnances

# Install the required libraries from a requirement file, using pypi
# Instala Unsloth compatível com CUDA 12.1 e Torch 2.3.0
pip install --upgrade pip setuptools wheel
#pip install "sympy>=1.12"
#pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
#pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
#pip install datasets pdf2image
pip install -r ../../requirements.txt

# Install Tesseract (binaries)
conda install -c conda-forge tesseract -y

# Install French data (manual download often more reliable on conda-forge)
TESSDATA_PREFIX=$CONDA_PREFIX/share/tessdata
wget https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata -P $TESSDATA_PREFIX/

which tesseract