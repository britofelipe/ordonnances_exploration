#!/bin/bash

#SBATCH --job-name=bert-ev2
#SBATCH --nodes=1
#SBATCH --partition=gpu_inter
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


python -u infer_one_image.py --input ordo_10.jpg --model_dir ./toobib-ordo-bert2bert-prod-100000 --lang fra --num_beams 4 --max_input_len 512  --max_target_len 512  --out_dir ./single_infer_out