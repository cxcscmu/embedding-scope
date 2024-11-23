#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:3
#SBATCH --mem=128G
#SBATCH --exclude=babel-1-31,babel-0-37,babel-15-32,babel-11-9
#SBATCH --array=0-3

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Evaluate the retrieval performance on the reconstructed embeddings.
##############################################################################

latentTopKPool=(32 64 128 256)
latentTopKPick=${latentTopKPool[$SLURM_ARRAY_TASK_ID]}

ENTRYPOINT="source.evaluator.retrieval.reconstructed"
SHAREDCMDS="--embedding miniCPM --dataset msMarco --indexGpuDevice 0 1"
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --latentTopK $latentTopKPick --retrieveTopK 100"
SHAREDCMDS="$SHAREDCMDS --modelName miniCPM-196K-$latentTopKPick --modelGpuDevice 2"
python3 -m $ENTRYPOINT $SHAREDCMDS
