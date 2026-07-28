# V-cycle Preconditioner for implicit_filter — Design

Date: 2026-07-28. Status: approved (plan review of the same date).
Source of truth for the algorithm: `/work/ab0995/a270225/neural_preconditioner`
(branch `npfilter-foundation`) — `VCYCLE_INTEGRATION_GUIDE.md`,
`src/npfilter/vcycle.py`, `tests/test_vcycle.py`.

## Problem

Production implicit_filter solves `A x = b` with Jacobi-preconditioned CG,
where `A = I + 2 (S/k²)ⁿ` and `S = D⁻¹K` (K symmetric PSD stencil, D the
lumped node/cell areas). In stiff regimes — biharmonic filtering (n=2) at
large filter-scale-to-resolution ratios — the spectrum spans ~10⁹–10¹⁰ and
Jacobi-CG needs thousands of iterations or fails outright (Danilov et al.
2024; reproduced in the 2026 preconditioner project on CORE2 at L ≤ 100 km).

## Decision summary (user-approved)

| Decision | Choice |
|---|---|
| Integration | Pure-JAX V-cycle apply as `M=` inside the **existing** `jax.scipy.sparse.linalg.cg`, solving the symmetrized SPD system `(D·A)x = D·b`. Setup host-side (numpy/scipy/pyamg), cached per filter instance. |
| Scope | `TriangularFilter._compute` (nodal + element systems) **and** `LatLonFilter._compute` (NemoFilter inherits). Excluded with clear errors: spatially varying k, `_compute_full` (metric-terms block system). |
| API | `Filter.set_preconditioner(preconditioner, **options)` / `get_preconditioner()`, mirroring `set_backend`. Choices: `'none'` (plain CG), `'jacobi'` (default, numerically untouched), `'vcycle'`. Runtime state, not persisted. |
| Dependencies | `pyamg` (+`scipy`) as optional extra `implicit_filter[vcycle]`, lazily imported, unpinned. |
| Benchmarks | Jacobi-CG vs V-cycle-CG on CORE2, ICON, NEMO; CPU and GPU; n∈{1,2}, L∈{50..1000} km; median-of-5; via SLURM (Levante, account ab0995). |

## Operator weighting (the load-bearing facts)

All multilevel algebra runs on `Â = D·A` (SPD); feeding raw `A` into SPD
machinery is the #1 error mode (S has ~33% relative asymmetry).

| System | Stencil attr | PSD convention | D |
|---|---|---|---|
| Triangular nodal | `_ss/_ii/_jj` | `S` as stored | `_area` |
| Triangular element | `_ss_e/_ii_e/_jj_e` | `S_e` as stored | `_elem_area` |
| Lat-lon / NEMO | `_ss/_ii/_jj` | **`−S`** (assembly is negative-semidefinite; solve uses `−1/k²` scaling) | `_area` |

Setup asserts symmetry of every level operator with a two-tier gate:
rel ≤ 1e-10 silent; ≤ 1e-6 warn-and-symmetrize (storage-precision roundoff —
caches saved by older package versions store the stencil in float32;
**measured 9.3e-9 on the 7.4M-node ICON cache**, which is therefore
supported); above 1e-6 **raises** — structural asymmetry. **Probe result
2026-07-28: the real NEMO/FOCI grid (366,480 points) measures 0.60** —
stretched latitude spacing makes `D·S` structurally asymmetric (no single
diagonal reweighting fixes both grid directions), so NEMO/stretched lat-lon
grids are rejected with a clear error and stay on Jacobi; uniform lat-lon
grids are supported (verified multilevel in tests).

## Algorithm (per the integration guide)

- Hierarchy: pyamg smoothed aggregation on `K = D·S`, k-independent, once
  per mesh/system; numpy global RNG saved/seeded/restored (determinism).
- Per (k, n): sparse `A`, `Â₀ = D·A`, Galerkin chain `Â_{l+1} = PᵀÂ_lP`
  with symmetry assertions; per-level `1/diag`; Chebyshev λmax by 30 seeded
  power iterations **×1.1 safety** (guide recommendation; deliberate
  deviation from the reference code, which omits it); dense Cholesky at the
  ~1000-unknown coarsest level (jitter fallback `1e-12·tr/n`).
- Apply: degree-3 Chebyshev pre/post smoothing per level + exact coarse
  solve; matvec-only, recursion unrolled at trace time → linear, symmetric,
  JAX-traceable; one V-cycle per CG iteration (≈ 8–10 fine matvec
  equivalents).
- Convergence: CG stops on the D-weighted residual, so the vcycle path adds
  a true-residual gate (unweighted `‖b−Ax‖/‖b‖ ≤ tol`, one retry at tol/10,
  then `SolverNotConvergedError`). JAX's cg returns no status — today
  non-convergence is silent; the gate fixes that on the new path only.

## Validation strategy

1. Stationary contraction of `x ← x + M(D(b−Ax))` — mutation-proven: the
   only test that catches sign-flip / mis-weighted / double-weighted M.
2. SPD of M (⟨Mu,v⟩=⟨u,Mv⟩, ⟨u,Mu⟩>0) — required for CG; absent in the
   FGMRES-based reference suite, added here.
3. Exactness vs `spsolve` (1e-7; 1e-6 at the stiff cell, cond-justified).
4. Failure-region regression (≤ 40 PCG iterations at n=2, L=500 km on the
   121-node fixture; ≥5× fewer than Jacobi).
5. Determinism (bit-identical hierarchy and setup across calls).
6. CPU↔GPU parity of full solves (rtol 1e-9), full suite run on a GPU node.
7. Default-path immutability: existing tests must pass byte-identically;
   diff audit confirms the jacobi branch is untouched.

## Non-goals / follow-ups

- Spatially varying k (needs `C^{-1/2}AC^{1/2}` similarity transform).
- Metric-terms coupled system (block symmetry analysis needed).
- Vectorizing `prepare` (superlinear; blocks >10⁶-node meshes) — known
  host-package issue, out of scope here.
