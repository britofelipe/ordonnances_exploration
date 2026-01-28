#!/bin/bash

#SBATCH --job-name=ordonnances
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/slurm-%j.out

echo "Running on $(hostname)"

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Load the conda environment
source activate ordonnances

# Display the installed packages using the conda install python interpreter
python -m pip list
which tesseract
#tesseract --list-langs
python generate_ordo_mimic.py --csv prescriptions_demo.csv --out output_mimic_fhir_ocr_template_demo --count 20 --blur 2.0 --jpeg 1.0 --stains 1.0 --skew 3.0
#python generate_ordo_mimic_segmented.py --csv prescriptions_demo.csv --out output_mimic_fhir_ocr_demo_segmented --start_idx 9000 --end_idx 9020 --count 20 --blur 2.0 --jpeg 1.0 --stains 0.0 --skew 3.0 #--only_train