#!/bin/bash
#SBATCH --job-name=controller
#SBATCH --partition=array
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=haok@andrew.cmu.edu
#SBATCH --array=0-135
#SBATCH --requeue

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

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 1 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 1 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 2 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 2 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 11 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 12 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 13 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 4 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 14 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 4 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 15 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 16 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 17 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 18 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 8 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 19 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 8 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 20 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 21 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 22 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 23 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 1 --feedbackAlpha 16 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 24 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 25 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 26 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 27 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 1 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 28 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 1 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 29 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 30 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 31 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 32 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 2 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 33 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 2 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 34 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 35 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 36 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 37 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 4 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 38 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 4 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 39 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 40 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 41 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 42 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 8 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 43 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 8 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 44 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 45 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 46 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 47 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 2 --feedbackAlpha 16 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 48 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 49 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 50 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 51 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 1 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 52 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 1 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 53 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 54 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 55 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 56 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 2 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 57 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 2 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 58 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 59 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 60 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 61 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 4 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 62 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 4 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 63 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 64 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 65 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 66 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 8 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 67 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 8 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 68 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 69 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 70 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 71 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 4 --feedbackAlpha 16 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 72 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 73 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 74 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 75 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 1 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 76 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 1 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 77 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 78 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 79 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 80 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 2 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 81 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 2 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 82 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 83 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 84 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 85 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 4 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 86 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 4 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 87 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 88 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 89 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 90 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 8 --feedbackDelta 0.32
elif [ $SLURM_ARRAY_TASK_ID -eq 91 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 8 --feedbackDelta 0.64

elif [ $SLURM_ARRAY_TASK_ID -eq 92 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 93 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 94 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 95 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 8 --feedbackAlpha 16 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 96 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 97 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 98 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 99 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 1 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 100 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 101 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 102 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 103 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 2 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 104 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 105 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 106 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 107 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 4 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 108 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 109 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 110 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 111 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 8 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 112 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 113 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 114 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 115 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 16 --feedbackAlpha 16 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 116 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 1 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 117 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 1 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 118 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 1 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 119 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 1 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 120 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 2 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 121 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 2 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 122 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 2 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 123 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 2 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 124 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 4 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 125 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 4 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 126 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 4 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 127 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 4 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 128 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 8 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 129 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 8 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 130 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 8 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 131 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 8 --feedbackDelta 0.32

elif [ $SLURM_ARRAY_TASK_ID -eq 132 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 16 --feedbackDelta 0.04
elif [ $SLURM_ARRAY_TASK_ID -eq 133 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 16 --feedbackDelta 0.08
elif [ $SLURM_ARRAY_TASK_ID -eq 134 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 16 --feedbackDelta 0.16
elif [ $SLURM_ARRAY_TASK_ID -eq 135 ]; then
    $ENTRYPOINT $SHAREDCMDS --feedbackTopK 32 --feedbackAlpha 16 --feedbackDelta 0.32
fi
