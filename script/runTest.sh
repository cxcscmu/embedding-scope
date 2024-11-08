#!/bin/bash
#SBATCH --job-name=runTest
#SBATCH --partition=general
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --exclude=babel-1-31,babel-0-37,babel-15-32

####################################################################
# Load the required modules.
####################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

####################################################################
# Run the test.
####################################################################

python3 -m pytest -v source
