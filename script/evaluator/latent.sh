#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=256G
#SBATCH --exclude=babel-1-31,babel-0-37,babel-15-32,babel-11-9

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Evaluate the retrieval performance on the reconstructed embeddings.
##############################################################################

SLURM_ARRAY_TASK_ID=3
latentTopKPool=(32 64 128 256)
latentTopKPick=${latentTopKPool[$SLURM_ARRAY_TASK_ID]}

# ENTRYPOINT="source.evaluator.retrieval.latent"
# SHAREDCMDS="--embedding miniCPM --dataset msMarco"
# SHAREDCMDS="$SHAREDCMDS --latentSize 196K --latentTopK $latentTopKPick --retrieveTopK 100"
# SHAREDCMDS="$SHAREDCMDS --modelName miniCPM-196K-$latentTopKPick --modelGpuDevice 0"
# python3 -m $ENTRYPOINT $SHAREDCMDS

# ENTRYPOINT="source.evaluator.retrieval.latent"
# SHAREDCMDS="--embedding bgeBase --dataset msMarco"
# SHAREDCMDS="$SHAREDCMDS --latentSize 196K --latentTopK $latentTopKPick --retrieveTopK 100"
# SHAREDCMDS="$SHAREDCMDS --modelName bgeBase-196K-$latentTopKPick --modelGpuDevice 0"
# python3 -m $ENTRYPOINT $SHAREDCMDS
