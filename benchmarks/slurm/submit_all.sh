#!/bin/bash
# Submit the full benchmark campaign: 3 meshes x 2 backends x 2
# preconditioners. CORE2 jobs wait for the one-off prepare job when its
# cache does not exist yet. Job IDs are appended to benchmarks/results/jobs.log.
set -euo pipefail
cd /work/ab0995/a270225/implicit_filter
mkdir -p benchmarks/results

log() { echo "$(date +%FT%T) $*" | tee -a benchmarks/results/jobs.log; }

DEP_CORE2=""
if [[ ! -f benchmarks/results/core2_cache.npz ]]; then
    # PREP_JOB lets an already-submitted prepare job be reused.
    PREP="${PREP_JOB:-$(sbatch --parsable benchmarks/slurm/prepare_core2.sbatch)}"
    log "prepare_core2 -> $PREP"
    DEP_CORE2="--dependency=afterok:$PREP"
fi

for MESH in core2 icon nemo; do
    for BACKEND in cpu gpu; do
        for PRECOND in jacobi vcycle; do
            ARGS=(--job-name "bench_${MESH}_${BACKEND}_${PRECOND}"
                  --export="ALL,MESH=${MESH},BACKEND=${BACKEND},PRECOND=${PRECOND}")
            if [[ $BACKEND == gpu ]]; then
                ARGS+=(-p gpu --gpus=1 --exclude=l40363 -t 04:00:00)
            else
                ARGS+=(-p shared -t 12:00:00)
            fi
            if [[ $MESH == icon ]]; then
                ARGS+=(--mem=100G)
                # 7.4M nodes: cap the CPU Jacobi iteration count so a DNC
                # cell costs bounded time (DNC is DNC at 5000 or 20000).
                [[ $BACKEND == cpu ]] && ARGS+=(--export="ALL,MESH=${MESH},BACKEND=${BACKEND},PRECOND=${PRECOND},MAXITER=5000")
            else
                ARGS+=(--mem=64G)
            fi
            [[ $MESH == core2 && -n $DEP_CORE2 ]] && ARGS+=($DEP_CORE2)
            JID=$(sbatch --parsable "${ARGS[@]}" benchmarks/slurm/bench.sbatch)
            log "bench_${MESH}_${BACKEND}_${PRECOND} -> $JID"
        done
    done
done
