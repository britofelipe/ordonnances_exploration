#!/bin/bash
#SBATCH --job-name=install-deps
#SBATCH --nodes=1
#SBATCH --partition=cpu_prod
#SBATCH --time=00:30:00
#SBATCH --output=./logs/install-%j.out
#SBATCH --mail-type=END,FAIL

source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
source activate ordonnances

echo "Installing dependencies..."
# Install training specific libraries
pip install peft bitsandbytes qwen-vl-utils torchvision
# Ensure PyTorch complies with requirements (already checked, but good to be safe)
# Upgrade transformers to ensure Qwen2-VL support
pip install git+https://github.com/huggingface/transformers.git@main --upgrade

echo "Installation complete."
