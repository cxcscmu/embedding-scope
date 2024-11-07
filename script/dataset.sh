#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --exclude=babel-1-31,babel-0-37
#SBATCH --array=0-7

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

# ENTRYPOINT="source.dataset.textRetrieval.msMarco"
# SHAREDCMDS="--numShards 4"
# python3 -m $ENTRYPOINT preparePassages $SHAREDCMDS

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding miniCPM --numShards 1024 --numWorkers $SLURM_ARRAY_TASK_COUNT --batchSize 128 --device 0"
python3 -m $ENTRYPOINT preparePassageEmbeddings $SHAREDCMDS --workerSeed $SLURM_ARRAY_TASK_ID
