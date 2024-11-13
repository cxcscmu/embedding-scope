#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:2
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
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --latentTopK 256 --retrieveTopK 128"
SHAREDCMDS="$SHAREDCMDS --modelName bgeBase-196K-256 --modelGpuDevice 1"

# Retrieve 16 passages for the initial query.
# For each retrieved passage, extract top-3 sparse features.
# Bump the query latent by 0.1 on dimensions with all the top-3 features.
# Reconstruct the query and retrieve 128 passages for evaluation.
$ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 3 --feedbackDelta 0.1
