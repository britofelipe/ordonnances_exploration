#!/bin/bash

# Define the email file path
EMAIL_FILE=".slurm_email"

# Check if email file exists
if [ ! -f "$EMAIL_FILE" ]; then
    echo "Erro: Arquivo '$EMAIL_FILE' não encontrado."
    exit 1
fi

# Read email from file
EMAIL=$(cat "$EMAIL_FILE")

# Check if email is not empty
if [ -z "$EMAIL" ]; then
    echo "Erro: Arquivo '$EMAIL_FILE' está vazio."
    exit 1
fi

# Check if script argument is provided
if [ -z "$1" ]; then
    echo "Uso: ./submit.sh <caminho_para_script_slurm>"
    exit 1
fi

# Determine the directory of the target script
SCRIPT_PATH="$1"
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

echo "Submitting job with email alerts to: $EMAIL"
echo "Target script: $SCRIPT_PATH"
echo "Working directory: $SCRIPT_DIR"

# Change to the script's directory to ensure relative paths work
cd "$SCRIPT_DIR" || exit 1

# Submit the job with email config, passing only the filename (since we cd'd)
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
sbatch --mail-user="$EMAIL" --mail-type=END,FAIL "$SCRIPT_NAME"
