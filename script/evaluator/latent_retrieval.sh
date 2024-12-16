#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --job-name=evaluator
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

topKPool=(32 64 128)
topKPick=${topKPool[$SLURM_ARRAY_TASK_ID]}

##############################################################################
# Dispatch the evaluation.
##############################################################################

entry="source.evaluator.latent_retrieval"

command="--dataset msMarco --embedding bgeBase"
command="$command --latentSize 196K --latentTopK $topKPick"
command="$command --modelName bgeBase-196K-$topKPick --modelDevice 0"
python3 -m $entry $command

command="--dataset msMarco --embedding miniCPM"
command="$command --latentSize 196K --latentTopK $topKPick"
command="$command --modelName miniCPM-196K-$topKPick --modelDevice 0"
python3 -m $entry $command
