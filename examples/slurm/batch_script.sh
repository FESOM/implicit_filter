#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100_80:4
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Activate your python enviroment
conda activate env-name

srun --label --cpu-bind=v --accel-bind=v python run_script.py

echo "Finished job."
sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
date
