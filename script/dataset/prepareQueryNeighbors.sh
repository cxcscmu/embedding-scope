#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=256G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --exclude=babel-1-31,babel-0-37,babel-15-32,babel-11-9

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding miniCPM --gpuDevice 0 1 2 3 --batchSize 512 --topK 128"
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition train
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition dev
