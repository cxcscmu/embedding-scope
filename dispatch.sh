#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

# Prepare the MS MARCO dataset.
# python3 -m source.dataset.msMarco getPassages
# python3 -m source.dataset.msMarco getQueries
