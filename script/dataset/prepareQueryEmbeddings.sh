#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --exclude=babel-1-31,babel-0-37
#SBATCH --array=0-7

####################################################################
# Load the required modules.
####################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

####################################################################
# Prepare the MS MARCO dataset.
####################################################################

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding miniCPM --gpuDevice 0 --batchSize 128 --workerCnt $SLURM_ARRAY_TASK_COUNT"
python3 -m $ENTRYPOINT prepareQueryEmbeddings $SHAREDCMDS --numShards 16 --partition "train" --workerIdx $SLURM_ARRAY_TASK_ID
python3 -m $ENTRYPOINT prepareQueryEmbeddings $SHAREDCMDS --numShards 2 --partition "dev" --workerIdx $SLURM_ARRAY_TASK_ID
python3 -m $ENTRYPOINT prepareQueryEmbeddings $SHAREDCMDS --numShards 2 --partition "eval" --workerIdx $SLURM_ARRAY_TASK_ID
