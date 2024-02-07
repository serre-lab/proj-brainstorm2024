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

# Email settings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sixuan_chen1@brown.edu

# Set up the environment by loading modules
#module load python3/3.8.5 or any other module if required

# Activate the virtual environment
module load miniconda3/23.11.0s
module load python/3.9.16s-x3wdtvt
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
#conda activate /users/schen336/.conda/envs/myenv
conda activate myenv

# Run a script using
nvidia-smi
lscpu
echo "WHICH Python:"
which python
python -c "import torch; print('TORCH_VERSION:',torch.__version__)"
python -u main.py -e 1000 -a 10000000
