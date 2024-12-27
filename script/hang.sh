#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=48:00:00
#SBATCH --job-name=hang
#SBATCH --partition=general

# This script is used to hang the job for 48 hours, while at the same time,
# the user may ssh into the compute node for interactive debugging.
sleep 48h
