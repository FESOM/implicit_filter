#!/bin/bash
#SBATCH --job-name=impfilt_prep_core2
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/ab0995/a270225/implicit_filter
#SBATCH --output=benchmarks/results/slurm_%x_%j.log
# One-off: prepare the CORE2 FESOM mesh (126,858 nodes; ~1 h, prepare is
# superlinear pure Python) and cache the filter for all benchmark runs.
source ~/.bashrc
conda activate env-name
set -euo pipefail
python - <<'EOF'
import time
from implicit_filter import FesomFilter

t0 = time.perf_counter()
f = FesomFilter()
f.prepare_from_file(
    "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/"
    "fesom.mesh.diag.nc")
f.save_to_file("benchmarks/results/core2_cache.npz")
print(f"prepared n2d={int(f._n2d)} e2d={int(f._e2d)} "
      f"in {time.perf_counter() - t0:.0f}s", flush=True)
EOF
