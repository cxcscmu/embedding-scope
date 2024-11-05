#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=256G

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

# entrypoint=source.dataset.msMarco
# python3 -m $entrypoint getPassages --numPartitions 2
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 0 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 1 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 2 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 3 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 4 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 5 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 6 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 7 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 8 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 9 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 10 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 11 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 12 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 13 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 14 --batchSize 128
# python3 -m $entrypoint getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 15 --batchSize 128
# python3 -m $entrypoint getQueries --numPartitions 1
# python3 -m $entrypoint getQueryEmbeddings --embedding MiniCPM --numPartitions 1 --partitionIndex 0 --batchSize 128
# python3 -m $entrypoint getRelevantPassages
# python3 -m $entrypoint getNeighborPassages --embedding MiniCPM --devices 0 1 2 3

##############################################################################
# Train the model.
##############################################################################

ENTRYPOINT="python3 -m source.trainer.v2410"
SHAREDCFGS="--dataset MsMarco --embedding MiniCPM --optimizer Adam --scheduler CosineAnnealing --devices 0 1"
$ENTRYPOINT $SHAREDCFGS --latentSize 196000 --latentTopK 32 --numNeighbors 8 --numEpochs 256 --learnRate 3e-4 --batchSize 256
