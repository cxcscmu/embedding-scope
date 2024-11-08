#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --exclude=babel-1-31,babel-0-37

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--numShards 4"
python3 -m $ENTRYPOINT preparePassages $SHAREDCMDS