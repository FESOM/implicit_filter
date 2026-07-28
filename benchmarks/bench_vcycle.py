"""Before/after benchmark: Jacobi-CG vs V-cycle-CG on real meshes, CPU and GPU.

Per (mesh, backend, preconditioner, n, L, tol) this records, following the
production perturbation-RHS convention (compute() solves A x' = b - A b):

- ``iters`` and ``relres`` from a jitted counted PCG identical in algorithm
  and stopping rule to the production ``jax.scipy`` CG;
- ``t_kernel`` (median of --repeats, first compiled call excluded,
  ``block_until_ready``): the pure solve kernel;
- ``t_e2e`` (median of --repeats, warm-up excluded): wall-clock of the
  public ``compute()`` call users experience (includes re-tracing; only at
  the production tolerance 1e-6, and only when the kernel converged within
  the benchmark iteration cap);
- V-cycle setup accounting: ``t_hierarchy`` (once per mesh, k-independent)
  and ``t_setup_kn`` (per (k, n)); ``matvec_equiv_per_iter``, the measured
  nnz-based cost of one V-cycle in units of one operator application, so
  iteration counts can be compared as work.

Results stream to --out as JSON, atomically rewritten after every record
(tmp + os.replace), with full provenance. Meshes whose stencil fails the
structural-symmetry gate (e.g. NEMO/stretched lat-lon grids) are recorded
as skipped for the V-cycle.

Usage (see benchmarks/slurm/ for the SLURM campaign):
  python benchmarks/bench_vcycle.py --mesh core2 --backend gpu \
      --precond vcycle --out benchmarks/results/bench_core2_gpu_vcycle.json
"""
import argparse
import json
import math
import os
import platform
import subprocess
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", choices=["core2", "icon", "nemo", "synthetic"],
                   required=True)
    p.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    p.add_argument("--precond", choices=["none", "jacobi", "vcycle"],
                   required=True)
    p.add_argument("--n", default="1,2")
    p.add_argument("--scales-km", default="50,100,200,500,1000")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--extra-tol", type=float, default=1e-9)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=20000)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_mesh(mesh):
    """Return (filter, family, field). field follows compute()'s data shape."""
    from implicit_filter import (FesomFilter, IconFilter, NemoFilter,
                                 TriangularFilter)

    if mesh == "core2":
        f = FesomFilter.load_from_file("benchmarks/results/core2_cache.npz")
        return f, "tri", real_or_random_field_tri(f, "data/u.fesom.1948.nc")
    if mesh == "icon":
        f = IconFilter.load_from_file("icon_cash.npz")
        return f, "tri", (random_field(int(f._n2d)), "synthetic:seed42")
    if mesh == "nemo":
        f = NemoFilter.load_from_file("nemo_cash.npz")
        return f, "latlon", real_or_random_field_nemo(
            f, "FOCI1.14-TEST.2020.04.1_1m_20110101_20111231_grid_T.nc")
    # synthetic: 41x41 structured mesh, 5 km spacing (login-safe smoke test)
    xs, ys = np.meshgrid(np.arange(41) * 5.0, np.arange(41) * 5.0)
    x, y = xs.ravel(), ys.ravel()
    tri = []
    for j in range(40):
        for i in range(40):
            n0 = j * 41 + i
            tri.append([n0, n0 + 1, n0 + 41])
            tri.append([n0 + 1, n0 + 42, n0 + 41])
    f = TriangularFilter()
    f.prepare(len(x), len(tri), np.array(tri), x, y, meshtype="m",
              cartesian=True)
    return f, "tri", (random_field(int(f._n2d)), "synthetic:seed42")


def random_field(size, seed=42):
    return np.random.default_rng(seed).normal(size=size)


def real_or_random_field_tri(f, path):
    n2d = int(f._n2d)
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        for name, var in ds.data_vars.items():
            arr = np.asarray(var.values)
            flat = arr.reshape(-1)
            if flat.size >= n2d and flat.size % n2d == 0:
                field = np.nan_to_num(flat[:n2d].astype(np.float64))
                if np.std(field) > 0:
                    return field, f"real:{os.path.basename(path)}:{name}"
    except Exception:
        pass
    return random_field(n2d), "synthetic:seed42"


def real_or_random_field_nemo(f, path):
    nx, ny = int(f._nx), int(f._ny)
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        for name in ["sosstsst", "tos", "sst", "votemper", "thetao"]:
            if name in ds:
                arr = np.asarray(ds[name].values)
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.shape in [(ny, nx), (nx, ny)]:
                    if arr.shape == (ny, nx):
                        arr = arr.T
                    return (np.nan_to_num(arr.astype(np.float64)),
                            f"real:{os.path.basename(path)}:{name}")
    except Exception:
        pass
    return random_field((nx, ny)), "synthetic:seed42"


def provenance(args):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return "?"
    import jax
    import implicit_filter
    return {
        "argv": vars(args),
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_rev": sh("module load git 2>/dev/null; git rev-parse --short HEAD"),
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "implicit_filter": getattr(implicit_filter, "__version__", "?"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class Flusher:
    def __init__(self, out, prov):
        self.out = out
        self.doc = {"provenance": prov, "records": []}

    def add(self, rec):
        self.doc["records"].append(rec)
        tmp = self.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.doc, fh, indent=1, default=float)
        os.replace(tmp, self.out)


def median(xs):
    return float(np.median(np.asarray(xs))) if xs else None


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import jax
    import jax.numpy as jnp
    import implicit_filter
    from implicit_filter.utils import _vcycle as vc

    # JAX resolves its platform set on first use; select the backend BEFORE
    # the mesh cache is loaded into device arrays (set_backend afterwards
    # would be a silent no-op for this process).
    implicit_filter.TriangularFilter().set_backend(args.backend)
    if args.backend == "gpu" and not any(
            d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU requested but no CUDA device visible")

    flt, family, (field, rhs_src) = load_mesh(args.mesh)

    prov = provenance(args)
    prov["rhs"] = rhs_src
    flush = Flusher(args.out, prov)

    field_flat = np.reshape(np.asarray(field, dtype=np.float64), -1)

    # PSD convention: negate the lat-lon stencil. With that, data_scaled =
    # ss * (1/k^2) matches the production scaling for both families, and so
    # do the operator and the Jacobi diagonal below.
    ss = np.asarray(flt._ss, dtype=np.float64)
    if family == "latlon":
        ss = -ss
    ii = np.asarray(flt._ii)
    jj = np.asarray(flt._jj)
    area = np.asarray(flt._area, dtype=np.float64)
    n_size = area.size

    opts = {"degree": 3, "alpha": 4.0, "n_cycles": 1, "max_levels": 6,
            "max_coarse": 1000, "seed": 42, "lam_safety": 1.1,
            "strength": "symmetric"}

    hierarchy, t_hier = None, None
    if args.precond == "vcycle":
        import scipy.sparse as sp
        S = sp.csr_matrix((ss, (ii, jj)), shape=(n_size, n_size))
        t0 = time.perf_counter()
        hierarchy = vc.build_hierarchy(S, area, max_levels=opts["max_levels"],
                                       max_coarse=opts["max_coarse"],
                                       seed=opts["seed"])
        t_hier = time.perf_counter() - t0
        sizes = [n_size] + [P.shape[1] for P in hierarchy]
        print(f"hierarchy {sizes} built in {t_hier:.2f}s", flush=True)

    ii_j, jj_j = jnp.asarray(ii), jnp.asarray(jj)
    area_j = jnp.asarray(area)

    tols = [args.tol] + ([args.extra_tol] if args.extra_tol else [])
    for n in [int(v) for v in args.n.split(",")]:
        for L in [float(v) for v in args.scales_km.split(",")]:
            k = 2 * math.pi / L
            data_scaled = jnp.asarray(ss * (1.0 / k**2))  # PSD convention

            def apply_A(x):
                y = x
                for _ in range(n):
                    y = jnp.zeros_like(x).at[ii_j].add(data_scaled * y[jj_j])
                return x + 2.0 * y

            u = jnp.asarray(field_flat)
            ttw = u - apply_A(u)

            rec_base = {"mesh": args.mesh, "backend": args.backend,
                        "precond": args.precond, "n": n, "L_km": L}

            if args.precond == "vcycle":
                import scipy.sparse as sp
                S = sp.csr_matrix((ss, (ii, jj)), shape=(n_size, n_size))
                try:
                    t0 = time.perf_counter()
                    data_kn = vc.setup_vcycle(
                        S, area, k, n, hierarchy, degree=opts["degree"],
                        alpha=opts["alpha"], n_cycles=opts["n_cycles"],
                        seed=opts["seed"], lam_safety=opts["lam_safety"])
                    t_setup = time.perf_counter() - t0
                except ValueError as e:
                    flush.add({**rec_base, "skipped": str(e)})
                    print(f"n={n} L={L}: SKIPPED (asymmetric stencil)",
                          flush=True)
                    continue
                M = vc.make_vcycle_preconditioner(data_kn)
                A_sys = lambda x: area_j * apply_A(x)
                b_sys = area_j * ttw
                # nnz-based work of one V-cycle in fine-operator applications.
                # Per smoothed level: pre-smooth with zero guess (degree-1
                # matvecs) + residual (1) + post-smooth (degree) = 2*degree
                # level matvecs; plus one R and one P application per level.
                # The +1 outside is CG's own operator application.
                nnz_apply = 2 * n * len(ss)
                nnz_lvls = [len(c[0]) for c in data_kn.A_coo]
                nnz_P = [len(c[0]) for c in data_kn.P_coo]
                cyc = (2 * opts["degree"]) * sum(nnz_lvls) + 2 * sum(nnz_P)
                matvec_equiv = 1.0 + cyc / nnz_apply
                extra = {"t_hierarchy": t_hier, "t_setup_kn": t_setup,
                         "sizes": list(data_kn.sizes),
                         "matvec_equiv_per_iter": matvec_equiv}
            else:
                diag_mask = ii == jj
                d1 = jnp.zeros(n_size).at[ii_j[diag_mask]].add(
                    data_scaled[diag_mask])
                approx_diag = 1.0 + 2.0 * d1 ** n
                M = (lambda r: r / approx_diag) if args.precond == "jacobi" \
                    else (lambda r: r)
                A_sys, b_sys = apply_A, ttw
                extra = {"matvec_equiv_per_iter": 1.0}

            for tol in tols:
                solve = jax.jit(
                    lambda b, _A=A_sys, _M=M, _t=tol:
                    vc.pcg_kernel(_A, b, _M, _t, args.maxiter))
                t0 = time.perf_counter()
                x, it = solve(b_sys)
                x.block_until_ready()
                t_first = time.perf_counter() - t0
                it = int(it)
                relres = float(jnp.linalg.norm(ttw - apply_A(x))
                               / jnp.linalg.norm(ttw))
                converged = relres <= tol and it < args.maxiter
                retried = False
                if (not converged and args.precond == "vcycle"
                        and it < args.maxiter):
                    # Mirror the production path's bounded retry: the kernel
                    # stops on the D-weighted residual, which can leave the
                    # unweighted one marginally above tol; production
                    # continues once at tol/10 before declaring failure.
                    retried = True
                    x, it2 = vc.pcg_kernel(A_sys, b_sys, M, tol * 0.1,
                                           args.maxiter, x)
                    x.block_until_ready()
                    it += int(it2)
                    relres = float(jnp.linalg.norm(ttw - apply_A(x))
                                   / jnp.linalg.norm(ttw))
                    converged = relres <= tol
                t_kernel = []
                if converged:
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        x2, it2 = solve(b_sys)
                        x2.block_until_ready()
                        t_kernel.append(time.perf_counter() - t0)
                t_e2e = []
                if (converged and tol == args.tol
                        and args.precond in ("jacobi", "vcycle")):
                    # set_preconditioner clears the instance cache, so it
                    # must not be re-issued per record (hierarchy rebuild).
                    if flt.get_preconditioner() != args.precond:
                        flt.set_preconditioner(args.precond)
                    field_in = np.asarray(field)
                    flt.compute(n, k, field_in)          # warm-up
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        flt.compute(n, k, field_in)
                        t_e2e.append(time.perf_counter() - t0)
                rec = {**rec_base, **extra, "tol": tol, "iters": it,
                       "relres": relres, "converged": converged,
                       "retried": retried,
                       "t_first": t_first, "t_kernel_med": median(t_kernel),
                       "t_kernel_runs": t_kernel,
                       "t_e2e_med": median(t_e2e), "t_e2e_runs": t_e2e}
                flush.add(rec)
                print(f"n={n} L={L:6.0f} tol={tol:.0e}: iters={it:6d} "
                      f"relres={relres:.2e} conv={converged} "
                      f"t_kernel={median(t_kernel)} t_e2e={median(t_e2e)}",
                      flush=True)


if __name__ == "__main__":
    main()
