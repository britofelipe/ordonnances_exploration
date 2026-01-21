#!/bin/bash
#SBATCH --job-name=vis-train
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=./logs/train-%j.out

echo "=========================================="
echo "Starting Visual Model Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Load environment
# Load environment
source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
source activate ordonnances

# Ensure logs dir exists
mkdir -p ./logs
mkdir -p ./checkpoints

# Launch Training
# Using Qwen2-VL-2B-Instruct for faster iteration.
# Change to Qwen/Qwen2-VL-7B-Instruct for final run.
# Paths are relative to visual/training/ because submit.sh cds there
python train_vision_model.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --data_dir ../dataset_v1 \
    --output_dir checkpoints/qwen2vl-layout-v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --save_steps 250 \
    --logging_steps 10

echo ""
echo "=========================================="
echo "Job Finished at $(date)"
echo "=========================================="
