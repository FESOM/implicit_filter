# Jacobi-CG vs V-cycle-CG: measured comparison

Solves of the production perturbation system `A x' = b − A·b` with `A = I + 2(S/k²)ⁿ`, `k = 2π/L`. Cells are `iterations / median kernel wall-clock` (jitted counted PCG identical to the production CG; first compiled call excluded; median of repeats; `block_until_ready`). DNC = did not converge within the iteration cap; *unsupported* = the mesh fails the V-cycle's structural symmetry gate (stretched lat-lon grids). One V-cycle iteration costs several operator applications — the measured factor is given per mesh below — so compare *work* via that factor, and wall-clock directly. — = cell not measured: the stiff biharmonic tail of the 7.4M-node ICON mesh is impractical on CPU for either method (≈5 s per Jacobi iteration, thousands of iterations; the GPU table covers those cells).

## core2

Hierarchy sizes: [126858, 11722, 735]; one V-cycle iteration ≈ 4.8 operator applications.
Setup cost: hierarchy 6.2 s once per mesh; ≈0.33 s per (k, n), cached.

### n=1, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 22 / 0.988 s | 4 / 1.93 s | 22 / 0.00182 s | 4 / 0.00261 s | 0.7× |
| 100 | 35 / 1.55 s | 6 / 2.75 s | 35 / 0.00281 s | 6 / 0.00357 s | 0.8× |
| 200 | 53 / 2.32 s | 7 / 3.07 s | 53 / 0.00415 s | 7 / 0.00408 s | 1.0× |
| 500 | 75 / 3.26 s | 9 / 3.03 s | 75 / 0.00583 s | 9 / 0.00407 s | 1.4× |
| 1000 | 117 / 5.08 s | 8 / 3.39 s | 117 / 0.00905 s | 8 / 0.00456 s | 2.0× |

### n=2, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 6643 / 581 s | 8 / 7.12 s | 6643 / 0.511 s | 8 / 0.0078 s | 65.5× |
| 100 | DNC (>20000) | 16 / 12.5 s | DNC (>20000) | 16 / 0.0136 s | — |
| 200 | 5579 / 484 s | 24 / 22 s | 3686 / 0.382 s | 24 / 0.0237 s | 16.1× |
| 500 | 10853 / 946 s | 40 / 36.7 s | 12644 / 0.783 s | 40 / 0.0365 s | 21.5× |
| 1000 | DNC (>20000) | 53 / 47.7 s | DNC (>20000) | 53 / 0.05 s | — |

### n=1, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 45 / 1.97 s | 6 / 2.71 s | 45 / 0.00358 s | 6 / 0.00357 s | 1.0× |
| 100 | 67 / 2.92 s | 9 / 3.84 s | 67 / 0.0052 s | 9 / 0.00502 s | 1.0× |
| 200 | 91 / 3.94 s | 11 / 4.58 s | 91 / 0.00707 s | 11 / 0.00604 s | 1.2× |
| 500 | 135 / 5.86 s | 12 / 4.91 s | 135 / 0.0104 s | 12 / 0.00652 s | 1.6× |
| 1000 | 220 / 9.53 s | 13 / 5.26 s | 220 / 0.0169 s | 13 / 0.00702 s | 2.4× |

### n=2, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 19191 / 1.76e+03 s | 10 / 9.8 s | 19191 / 1.48 s | 10 / 0.0107 s | 138.2× |
| 100 | DNC (>20000) | 20 / 18.5 s | DNC (>20000) | 20 / 0.0204 s | — |
| 200 | DNC (>20000) | 37 / 34.1 s | 17988 / 1.3 s | 37 / 0.0365 s | 35.6× |
| 500 | DNC (>20000) | 70 / 62.9 s | 18174 / 1.54 s | 70 / 0.0628 s | 24.5× |
| 1000 | DNC (>20000) | 115 / 102 s | DNC (>20000) | 115 / 0.103 s | — |

### End-to-end `compute()` wall-clock (tol 1e-6, includes tracing/setup-cache hits)

| n | L (km) | backend | Jacobi | V-cycle |
|---:|---:|---|---:|---:|
| 1 | 50 | cpu | 1.06 s | 2.37 s |
| 1 | 50 | gpu | 0.225 s | 0.72 s |
| 1 | 100 | cpu | 1.62 s | 3.45 s |
| 1 | 100 | gpu | 0.227 s | 0.721 s |
| 1 | 200 | cpu | 2.39 s | 3.63 s |
| 1 | 200 | gpu | 0.227 s | 0.731 s |
| 1 | 500 | cpu | 3.34 s | 5.08 s |
| 1 | 500 | gpu | 0.229 s | 1.42 s |
| 1 | 1000 | cpu | 5.14 s | 3.81 s |
| 1 | 1000 | gpu | 0.233 s | 0.727 s |
| 2 | 50 | cpu | 574 s | 10.1 s |
| 2 | 50 | gpu | 0.799 s | 1.46 s |
| 2 | 100 | cpu | — | 17.2 s |
| 2 | 100 | gpu | — | 1.48 s |
| 2 | 200 | cpu | 673 s | 22.7 s |
| 2 | 200 | gpu | 0.649 s | 0.764 s |
| 2 | 500 | cpu | 987 s | 36.6 s |
| 2 | 500 | gpu | 0.936 s | 0.782 s |
| 2 | 1000 | cpu | — | 48 s |
| 2 | 1000 | gpu | — | 0.785 s |

## icon

Hierarchy sizes: [7487687, 823133, 50603, 1982, 113]; one V-cycle iteration ≈ 5.0 operator applications.
Setup cost: hierarchy 29 s once per mesh; ≈14 s per (k, n), cached.

### n=1, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 22 / 59 s | 4 / 90.9 s | 22 / 0.0392 s | 4 / 0.0519 s | 0.8× |
| 100 | 38 / 102 s | 4 / 113 s | 38 / 0.0651 s | 4 / 0.0628 s | 1.0× |
| 200 | 65 / 170 s | 5 / 136 s | 65 / 0.11 s | 5 / 0.0736 s | 1.5× |
| 500 | 117 / 303 s | 6 / 159 s | 117 / 0.196 s | 6 / 0.0872 s | 2.2× |
| 1000 | 173 / 447 s | 7 / 181 s | 173 / 0.288 s | 7 / 0.098 s | 2.9× |

### n=2, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 66 / 354 s | 10 / 573 s | 66 / 0.179 s | 10 / 0.421 s | 0.4× |
| 100 | 166 / 851 s | 15 / 836 s | 167 / 0.448 s | 15 / 0.613 s | 0.7× |
| 200 | 423 / 2.15e+03 s | 24 / 1.31e+03 s | 423 / 1.13 s | 24 / 0.957 s | 1.2× |
| 500 | — | — | 885 / 2.38 s | 42 / 1.65 s | 1.4× |
| 1000 | — | — | 1010 / 2.69 s | 60 / 2.33 s | 1.2× |

### n=1, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 38 / 100 s | 6 / 158 s | 38 / 0.065 s | 6 / 0.0858 s | 0.8× |
| 100 | 67 / 175 s | 7 / 181 s | 67 / 0.113 s | 7 / 0.098 s | 1.2× |
| 200 | 120 / 313 s | 8 / 204 s | 120 / 0.201 s | 8 / 0.11 s | 1.8× |
| 500 | 254 / 656 s | 11 / 273 s | 254 / 0.423 s | 11 / 0.152 s | 2.8× |
| 1000 | 444 / 1.15e+03 s | 12 / 295 s | 444 / 0.737 s | 12 / 0.159 s | 4.6× |

### n=2, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 123 / 654 s | 17 / 939 s | 123 / 0.331 s | 17 / 0.689 s | 0.5× |
| 100 | 395 / 2.07e+03 s | 27 / 1.46e+03 s | 376 / 1.05 s | 27 / 1.07 s | 1.0× |
| 200 | — | 47 / 2.53e+03 s | 1177 / 3.15 s | 47 / 1.84 s | 1.7× |
| 500 | — | — | 4631 / 12.3 s | 99 / 3.83 s | 3.2× |
| 1000 | — | — | 12024 / 32 s | 188 / 7.23 s | 4.4× |

### End-to-end `compute()` wall-clock (tol 1e-6, includes tracing/setup-cache hits)

| n | L (km) | backend | Jacobi | V-cycle |
|---:|---:|---|---:|---:|
| 1 | 50 | cpu | 64.1 s | 151 s |
| 1 | 50 | gpu | 0.796 s | 2.93 s |
| 1 | 100 | cpu | 105 s | 123 s |
| 1 | 100 | gpu | 0.817 s | 1.69 s |
| 1 | 200 | cpu | 177 s | 146 s |
| 1 | 200 | gpu | 0.86 s | 1.73 s |
| 1 | 500 | cpu | 309 s | 169 s |
| 1 | 500 | gpu | 0.944 s | 1.73 s |
| 1 | 1000 | cpu | 451 s | 190 s |
| 1 | 1000 | gpu | 1.04 s | 1.72 s |
| 2 | 50 | cpu | 349 s | 593 s |
| 2 | 50 | gpu | 0.947 s | 2.11 s |
| 2 | 100 | cpu | 866 s | 852 s |
| 2 | 100 | gpu | 1.22 s | 2.31 s |
| 2 | 200 | cpu | 2.18e+03 s | 1.33e+03 s |
| 2 | 200 | gpu | 1.92 s | 2.64 s |
| 2 | 500 | gpu | 3.18 s | 3.34 s |
| 2 | 1000 | gpu | 3.54 s | 3.95 s |

## nemo

### n=1, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 15 / 0.922 s | unsupported | 15 / 0.00154 s | unsupported | — |
| 100 | 25 / 1.5 s | unsupported | 25 / 0.00246 s | unsupported | — |
| 200 | 43 / 2.55 s | unsupported | 43 / 0.00409 s | unsupported | — |
| 500 | 81 / 4.78 s | unsupported | 81 / 0.00758 s | unsupported | — |
| 1000 | 123 / 7.15 s | unsupported | 123 / 0.0114 s | unsupported | — |

### n=2, production tolerance 1e-6

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 25 / 3.01 s | unsupported | 25 / 0.00311 s | unsupported | — |
| 100 | 78 / 9.26 s | unsupported | 78 / 0.00944 s | unsupported | — |
| 200 | 216 / 25 s | unsupported | 216 / 0.0258 s | unsupported | — |
| 500 | 647 / 73.9 s | unsupported | 647 / 0.0645 s | unsupported | — |
| 1000 | 969 / 108 s | unsupported | 969 / 0.0966 s | unsupported | — |

### n=1, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 23 / 1.41 s | unsupported | 23 / 0.00227 s | unsupported | — |
| 100 | 41 / 2.42 s | unsupported | 41 / 0.00394 s | unsupported | — |
| 200 | 75 / 4.44 s | unsupported | 75 / 0.00702 s | unsupported | — |
| 500 | 162 / 9.41 s | unsupported | 162 / 0.015 s | unsupported | — |
| 1000 | 298 / 17.3 s | unsupported | 298 / 0.0275 s | unsupported | — |

### n=2, strict tolerance 1e-9

| L (km) | CPU Jacobi | CPU V-cycle | GPU Jacobi | GPU V-cycle | GPU speedup |
|---:|---|---|---|---|---:|
| 50 | 40 / 4.99 s | unsupported | 40 / 0.00489 s | unsupported | — |
| 100 | 135 / 15.8 s | unsupported | 135 / 0.0161 s | unsupported | — |
| 200 | 438 / 50.3 s | unsupported | 438 / 0.0439 s | unsupported | — |
| 500 | 2130 / 243 s | unsupported | 2130 / 0.213 s | unsupported | — |
| 1000 | 9279 / 1.02e+03 s | unsupported | 9279 / 0.932 s | unsupported | — |

### End-to-end `compute()` wall-clock (tol 1e-6, includes tracing/setup-cache hits)

| n | L (km) | backend | Jacobi | V-cycle |
|---:|---:|---|---:|---:|
| 1 | 50 | cpu | 1.04 s | — |
| 1 | 50 | gpu | 0.282 s | — |
| 1 | 100 | cpu | 1.62 s | — |
| 1 | 100 | gpu | 0.283 s | — |
| 1 | 200 | cpu | 2.66 s | — |
| 1 | 200 | gpu | 0.285 s | — |
| 1 | 500 | cpu | 4.85 s | — |
| 1 | 500 | gpu | 0.288 s | — |
| 1 | 1000 | cpu | 7.35 s | — |
| 1 | 1000 | gpu | 0.294 s | — |
| 2 | 50 | cpu | 3.21 s | — |
| 2 | 50 | gpu | 0.305 s | — |
| 2 | 100 | cpu | 9.26 s | — |
| 2 | 100 | gpu | 0.312 s | — |
| 2 | 200 | cpu | 25.2 s | — |
| 2 | 200 | gpu | 0.329 s | — |
| 2 | 500 | cpu | 74 s | — |
| 2 | 500 | gpu | 0.382 s | — |
| 2 | 1000 | cpu | 107 s | — |
| 2 | 1000 | gpu | 0.418 s | — |

## Provenance

- `bench_core2_cpu_jacobi.json`: host l40002.lvt.dkrz.de, job 26522099, git 668fefd, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T16:48:16
- `bench_core2_cpu_jacobi_tail.json`: host l40020.lvt.dkrz.de, job 26536705, git 5d1ef7a, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-29T11:30:10
- `bench_core2_cpu_vcycle.json`: host l40001.lvt.dkrz.de, job 26522570, git 92704f7, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T17:11:18
- `bench_core2_gpu_jacobi.json`: host l40360.lvt.dkrz.de, job 26522101, git 668fefd, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T17:01:44
- `bench_core2_gpu_vcycle.json`: host l50036.lvt.dkrz.de, job 26522571, git 92704f7, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T17:09:04
- `bench_icon_cpu_jacobi.json`: host l40020.lvt.dkrz.de, job 26522103, git 668fefd, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T16:59:01
- `bench_icon_cpu_jacobi_tail.json`: host l40015.lvt.dkrz.de, job 26536706, git 5d1ef7a, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-29T12:13:28
- `bench_icon_cpu_vcycle.json`: host l40006.lvt.dkrz.de, job 26523028, git 5d1ef7a, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T17:48:41
- `bench_icon_cpu_vcycle_tail.json`: host l40010.lvt.dkrz.de, job 26536707, git 5d1ef7a, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-29T12:29:10
- `bench_icon_gpu_jacobi.json`: host l40360.lvt.dkrz.de, job 26522107, git 5782b3d, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T16:44:29
- `bench_icon_gpu_vcycle.json`: host l40360.lvt.dkrz.de, job 26523029, git 5d1ef7a, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T17:42:41
- `bench_nemo_cpu_jacobi.json`: host l40010.lvt.dkrz.de, job 26522110, git 668fefd, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T17:01:54
- `bench_nemo_cpu_vcycle.json`: host l40008.lvt.dkrz.de, job 26522575, git 92704f7, jax 0.5.2, devices ['TFRT_CPU_0'], rhs synthetic:seed42, 2026-07-28T17:39:06
- `bench_nemo_gpu_jacobi.json`: host l40360.lvt.dkrz.de, job 26522112, git 92704f7, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T17:09:06
- `bench_nemo_gpu_vcycle.json`: host l50036.lvt.dkrz.de, job 26522576, git 92704f7, jax 0.5.2, devices ['cuda:0'], rhs synthetic:seed42, 2026-07-28T17:13:51
