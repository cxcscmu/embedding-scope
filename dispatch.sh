#!/bin/bash
#SBATCH --job-name=scope
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

# python3 -m source.dataset.msMarco getPassages --numPartitions 2
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 0 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 1 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 2 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 3 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 4 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 5 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 6 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 7 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 8 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 9 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 10 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 11 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 12 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 13 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 14 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding MiniCPM --numPartitions 16 --partitionIndex 15 --batchSize 128
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 0 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 1 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 2 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 3 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 4 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 5 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 6 --batchSize 512
# python3 -m source.dataset.msMarco getPassageEmbeddings --embedding BgeBase --numPartitions 8 --partitionIndex 7 --batchSize 512
# python3 -m source.dataset.msMarco getQueries --numPartitions 1
# python3 -m source.dataset.msMarco getQueryEmbeddings --embedding MiniCPM --numPartitions 1 --partitionIndex 0 --batchSize 128
# python3 -m source.dataset.msMarco getQueryEmbeddings --embedding BgeBase --numPartitions 1 --partitionIndex 0 --batchSize 512
