#!/bin/bash

# Request runtime:
#SBATCH --time=96:00:00

# Ask for the batch partition
#SBATCH -p gpu --gres=gpu:1


# Use memory (CPU RAM):
#SBATCH --mem=64G

# Specify a job name using modevalue:
#SBATCH -J trainingBrainstorm
#SBATCH -N 1
# Specify an output file using modevalue
#SBATCH -o outfile.out
#SBATCH -e errfile.err
#SBATCH --account=carney-tserre-condo

# Email settings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sixuan_chen1@brown.edu

# Set up the environment by loading modules
#module load python3/3.8.5 or any other module if required

# Activate the virtual environment
module load miniconda3/23.11.0s
module load python/3.9.16s-x3wdtvt
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate /users/schen336/.conda/envs/myenv

# Run a script using
nvidia-smi
lscpu
#python -u main.py 
python dataset/dataset4individual_Clean.py
