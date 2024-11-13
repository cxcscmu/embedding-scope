#!/bin/bash
#SBATCH --job-name=controller
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
# Perform the baseline retrieval on BgeBase autoencoder.
##############################################################################

ENTRYPOINT="source.controller.baseline"
SHAREDCMDS="--embedding bgeBase --dataset msMarco --indexGpuDevice 0"
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --latentTopK 256 --retrieveTopK 128"
SHAREDCMDS="$SHAREDCMDS --modelName bgeBase-196K-256 --modelGpuDevice 1"
python3 -m $ENTRYPOINT $SHAREDCMDS
