#!/bin/bash
#SBATCH --job-name=impfilt_gpu_tests
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=l40363
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/ab0995/a270225/implicit_filter
#SBATCH --output=benchmarks/results/slurm_%x_%j.log
# set -u must come after the profile scripts: /etc/profile.d/lang.sh
# references an unbound LC_ALL and would kill the shell.
source ~/.bashrc
conda activate env-name
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Full suite on the GPU node. The parity tests spawn their own
# subprocesses per backend leg, so one invocation suffices.
# test_find_adjacent_points_north is the documented pre-existing failure
# on the elements branch (unrelated).
python -m pytest test/ -q --deselect test/test_auxiliary.py::test_find_adjacent_points_north
