#!/bin/bash

#SBATCH --job-name=ordonnances
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00
#SBATCH --output=/logs/slurm-%j.out

echo "Running on $(hostname)"

# Load the conda module
module load anaconda3/2022.10/gcc-13.1.0

# Load the conda environment
source activate ordonnances

# Display the installed packages using the conda install python interpreter
python -m pip list
python generate_ordo_mimic.py --csv prescriptions_demo.csv --out output_mimic_fhir --count 20 --blur 2.0 --jpeg 1.0 --stains 1.0 --skew 3.0