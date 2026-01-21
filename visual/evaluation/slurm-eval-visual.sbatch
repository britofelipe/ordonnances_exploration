#!/bin/bash
#SBATCH --job-name=vis-eval
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/eval-%j.out

echo "=========================================="
echo "Starting Visual Model Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
source activate ordonnances

mkdir -p ./logs
mkdir -p ./results

# Path to the specific checkpoint or final model
# The training script saves the final model to 'visual/checkpoints/qwen2vl-layout-v1'
ADAPTER_PATH="../checkpoints/qwen2vl-layout-v1"

# Check if adapter exists, else warn
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "WARNING: Adapter path $ADAPTER_PATH does not exist yet. Using base model only (Baseline)."
    # You might want to exit here if you strictly need the adapter
    # exit 1
fi

python eval_vision_rag.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --adapter_path "$ADAPTER_PATH" \
    --data_dir ../generation/dataset_v1 \
    --output_dir ./results \
    --test_count 50

echo "Evaluation Finished"
