#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --exclude=babel-1-31,babel-0-37,babel-15-32,babel-11-9
#SBATCH --array=0-3

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Train the autoencoder with MiniCPM.
##############################################################################

# ENTRYPOINT="python3 -m source.trainer.v2410"
# SHAREDCMDS="--embedding miniCPM --dataset msMarco"
# SHAREDCMDS="$SHAREDCMDS --latentSize 196K --nearbyTopK 8"
# SHAREDCMDS="$SHAREDCMDS --optimizer Adam --scheduler CosineAnnealing"
# SHAREDCMDS="$SHAREDCMDS --learningRate 3e-4 --numEpochs 256 --batchSize 256"

# latentTopkPool=(32 64 128 256)
# latentTopKPick=${latentTopkPool[$SLURM_ARRAY_TASK_ID]}
# $ENTRYPOINT $SHAREDCMDS --name "miniCPM-196K-$latentTopKPick" --latentTopK $latentTopKPick

##############################################################################
# Train the autoencoder with BgeBase.
##############################################################################

ENTRYPOINT="python3 -m source.trainer.v2410"
SHAREDCMDS="--embedding bgeBase --dataset msMarco"
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --nearbyTopK 8"
SHAREDCMDS="$SHAREDCMDS --optimizer Adam --scheduler CosineAnnealing"
SHAREDCMDS="$SHAREDCMDS --learningRate 3e-4 --numEpochs 256 --batchSize 256"

latentTopkPool=(32 64 128 256)
latentTopKPick=${latentTopkPool[$SLURM_ARRAY_TASK_ID]}
$ENTRYPOINT $SHAREDCMDS --name "bgeBase-196K-$latentTopKPick" --latentTopK $latentTopKPick
