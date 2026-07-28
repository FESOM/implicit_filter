#!/bin/bash
#SBATCH --job-name=impfilt_bench
#SBATCH --account=ab0995
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/work/ab0995/a270225/implicit_filter
#SBATCH --output=benchmarks/results/slurm_%x_%j.log
# Parameterized benchmark job. Partition/GPU/mem/time come from the sbatch
# command line (see submit_all.sh); MESH/BACKEND/PRECOND/MAXITER via --export.
source ~/.bashrc
conda activate env-name
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python benchmarks/bench_vcycle.py \
    --mesh "${MESH}" --backend "${BACKEND}" --precond "${PRECOND}" \
    --maxiter "${MAXITER:-20000}" \
    --out "benchmarks/results/bench_${MESH}_${BACKEND}_${PRECOND}.json"
