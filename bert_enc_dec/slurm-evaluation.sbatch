#!/bin/bash

#SBATCH --job-name=bert-ev2
#SBATCH --nodes=1
#SBATCH --partition=cpu_inter
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"

# Ensure module is visible 
source /etc/profile

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Create conda environment 
# and prevent the use of system site-packages
# that will interfere with the conda env
export PYTHONNOUSERSITE=1

# Activate the environment
source activate bert_env

# ==========================================
# FIX INSTALLATION (Run this once)
# ==========================================
# 1. Uninstall the existing incompatible torch version
#echo "Uninstalling old torch..."
#pip uninstall -y torch torchvision torchaudio

# 2. Install the version compatible with GTX 1080 Ti (CUDA 11.8)
#echo "Installing Torch for CUDA 11.8..."
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. (Optional) Ensure dependencies are present
# pip install datasets transformers 'accelerate>=0.26.0'

# ==========================================
# DIAGNOSTICS
# ==========================================
#echo "--- GPU Status (nvidia-smi) ---"
#nvidia-smi

#echo "--- Python Torch Check ---"
#python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

#echo "--- Pip List ---"
#python -m pip list | grep torch

# ==========================================
# RUN EVALUATION
# ==========================================
#echo "--- Starting Benchmark ---"
#pip freeze | egrep 'transformers|torch|tokenizers|sentencepiece|safetensors'

python bert_evaluation_benchmark.py