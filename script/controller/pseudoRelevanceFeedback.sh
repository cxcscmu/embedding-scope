#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:3
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Perform the pseudo-relevance feedback on BgeBase autoencoder.
##############################################################################

ENTRYPOINT="python3 -m source.controller.pseudoRelevanceFeedback"
SHAREDCMDS="--embedding bgeBase --dataset msMarco --indexGpuDevice 0"
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --modelGpuDevice 1"
SHAREDCMDS="$SHAREDCMDS --feedbackTopK 10 --retrieveTopK 128"
SHAREDCMDS="$SHAREDCMDS --feedbackAlpha 1 --feedbackBeta 0.1"

SLURM_ARRAY_TASK_ID=0
latentTopkPool=(32 64 128 256)
latentTopKPick=${latentTopkPool[$SLURM_ARRAY_TASK_ID]}
$ENTRYPOINT $SHAREDCMDS --modelName "bgeBase-196K-$latentTopKPick" --latentTopK $latentTopKPick
