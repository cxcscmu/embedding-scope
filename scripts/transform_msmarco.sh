#!/bin/bash
#SBATCH --nodes=4                    # Request 4 compute nodes
#SBATCH --ntasks-per-node=2          # Request 2 tasks per node
#SBATCH --cpus-per-task=2            # Request 2 CPU cores per task
#SBATCH --mem=32GB                   # Request 32GB RAM per node
#SBATCH --time=00:10:00              # Set a 10-minute time limit
#SBATCH --partition=general          # Set the partition
#SBATCH --job-name=transform_msmarco # Set the name of the job
#SBATCH --output=logs/slurm-%j.out   # Set the output file

source devconfig.env
source ~/miniconda3/etc/profile.d/conda.sh
ls -1t logs/slurm-*.out | tail -n +11 | xargs rm -f

conda activate scope
ENTRY=sources.dataset.transform_msmarco

if [ -z $SLURM_JOBID ]; then 
    python3 -m $ENTRY
else 
    srun -W 0 python3 -m $ENTRY
fi
