#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --job-name=trainer
#SBATCH --partition=general
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --array=0-2

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Assign the current task.
##############################################################################

latentTopkPool=(32 64 128)
latentTopKPick=${latentTopkPool[$SLURM_ARRAY_TASK_ID]}

##############################################################################
# Dispatch the training script.
##############################################################################

ENTRYPOINT="source.trainer.v2410"

SHAREDCMDS="--embedding bgeBase --dataset msMarco"
SHAREDCMDS="$SHAREDCMDS --latentSize 196K --nearbyTopK 8"
SHAREDCMDS="$SHAREDCMDS --optimizer Adam --scheduler CosineAnnealing"
SHAREDCMDS="$SHAREDCMDS --learningRate 1e-3 --numEpochs 96 --batchSize 512"
python3 -m $ENTRYPOINT $SHAREDCMDS --name "bgeBase-196K-$latentTopKPick" --latentTopK $latentTopKPick
