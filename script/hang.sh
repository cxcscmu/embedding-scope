#!/bin/bash
#SBATCH --job-name=hang
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=96G

# This script is used to hang the job for 48 hours, while at the same time,
# the user may ssh into the compute node for interactive debugging.
sleep 48h
