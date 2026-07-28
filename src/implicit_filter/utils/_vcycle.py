"""Symmetric geometric V-cycle preconditioner for the implicit filter CG solve.

Ports the reference implementation from the 2026 preconditioner project
(neural_preconditioner, branch ``npfilter-foundation``,
``src/npfilter/vcycle.py``) following ``VCYCLE_INTEGRATION_GUIDE.md``. All
multilevel algebra runs on the symmetrized SPD operator ``A_hat = D @ A``
where ``A = I + 2 (S/k^2)^n`` and ``S = D^{-1} K`` with symmetric ``K``;
feeding raw ``A`` into this machinery is a bug (S has ~33% relative
asymmetry on real meshes).

Setup (host side, numpy/scipy/pyamg): smoothed-aggregation transfer
operators (k-independent, once per mesh/system), then per (k, n): Galerkin
coarse chain with roundoff-symmetry assertions, Chebyshev spectral bounds
via seeded power iteration (x1.1 safety), dense Cholesky on the coarsest
level.

Apply: matvec-only JAX closure (Chebyshev(3) pre/post smoothing, exact
coarse solve) -- traceable inside ``jax.scipy.sparse.linalg.cg``; identical
on CPU and GPU.
"""
import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# Two-tier symmetry gate for A_hat = D*A. Below _SYM_RTOL_CLEAN the
# asymmetry is float64 roundoff of the Galerkin products (silent). Between
# the tiers it is storage-precision roundoff -- filter caches saved by older
# package versions hold the stencil in float32 (measured 9.3e-9 on the 7.4M
# ICON cache) -- harmless for a preconditioner, so warn and symmetrize.
# Above _SYM_RTOL_HARD the operator is structurally asymmetric (wrong D
# side, stretched lat-lon grid: measured 0.6 on NEMO/FOCI) and proceeding
# would hide broken numerics.
_SYM_RTOL_CLEAN = 1e-10
_SYM_RTOL_HARD = 1e-6
_POWER_ITERS = 30


def _require_pyamg():
    """Import pyamg lazily so the core package works without it."""
    try:
        import pyamg
    except ImportError as e:
        raise ImportError(
            "The V-cycle preconditioner requires pyamg for hierarchy setup. "
            "Install it with: pip install 'implicit_filter[vcycle]'"
        ) from e
    return pyamg


def build_hierarchy(S, area, *, max_levels=6, max_coarse=1000,
                    strength="symmetric", seed=42):
    """Build prolongation operators by smoothed aggregation on ``K = D @ S``.

    The hierarchy depends only on the mesh graph, not on the filter scale
    ``k`` or order ``n``: the same transfer operators serve every (k, n).

    Parameters
    ----------
    S : scipy.sparse matrix
        Area-normalized stencil ``D^{-1} K`` in the PSD convention
        (symmetric K, positive diagonal).
    area : np.ndarray
        Lumped node/cell areas (the diagonal of D).
    max_levels : int, optional
        Maximum number of levels in the hierarchy.
    max_coarse : int, optional
        Stop coarsening once a level has at most this many unknowns; that
        level is solved directly (dense Cholesky) in the V-cycle.
    strength : str, optional
        pyamg strength-of-connection measure.
    seed : int, optional
        pyamg draws from numpy's *global* random state; it is saved, seeded
        with this value and restored, so hierarchies are bit-reproducible.

    Returns
    -------
    list of scipy.sparse.csr_matrix
        Prolongators ``P_l`` (fine to coarse, ``R = P.T``); empty when the
        system already has at most ``max_coarse`` unknowns.
    """
    pyamg = _require_pyamg()
    K = sp.diags(np.asarray(area, dtype=np.float64)) @ sp.csr_matrix(
        S, dtype=np.float64)
    if K.shape[0] <= max_coarse:
        return []
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        ml = pyamg.smoothed_aggregation_solver(
            sp.csr_matrix(K), max_levels=max_levels, max_coarse=max_coarse,
            strength=strength)
    finally:
        np.random.set_state(rng_state)
    return [sp.csr_matrix(lvl.P, dtype=np.float64) for lvl in ml.levels[:-1]]


@dataclass(frozen=True)
class VcycleData:
    """Per-(k, n) V-cycle state. Levels: 0 = finest .. L = coarsest (dense).

    ``A_coo``/``P_coo``/``R_coo`` hold per-level operators as JAX COO
    triplets ``(ss, ii, jj, n_rows)``; the coarsest level is a dense
    Cholesky factor instead. ``inv_diags``/``lam_max`` cover the smoothed
    levels only (the coarsest is solved exactly, never smoothed).
    """
    A_coo: Tuple
    P_coo: Tuple
    R_coo: Tuple
    inv_diags: Tuple
    lam_max: Tuple
    coarse_chol: jnp.ndarray
    coarse_lower: bool
    sizes: Tuple
    degree: int
    alpha: float
    n_cycles: int


def filter_matrix(S, k, n):
    """``A = I + 2 (S / k^2)^n`` as a scipy CSR matrix (true sparse power)."""
    A1 = sp.csr_matrix(S, dtype=np.float64) * (1.0 / float(k) ** 2)
    A = sp.identity(S.shape[0], format="csr", dtype=np.float64) + 2.0 * (A1 ** int(n))
    A.eliminate_zeros()
    return A.tocsr()


def _assert_roundoff_symmetric(A, label):
    """Return the symmetrized matrix, refusing structural asymmetry.

    Roundoff-scale asymmetry is averaged away (silently below
    ``_SYM_RTOL_CLEAN``, with a warning up to ``_SYM_RTOL_HARD`` -- the
    storage-precision tier for float32-saved filter caches). Anything above
    the hard gate means the fine-level weighting is wrong (D on the wrong
    side) or the mesh/stencil is unsupported, and silently symmetrizing
    would hide a broken preconditioner.
    """
    diff = (A - A.T).tocoo()
    denom = max(np.abs(A.data).max(), 1e-300)
    rel = (np.abs(diff.data).max() / denom) if diff.nnz else 0.0
    if rel > _SYM_RTOL_HARD:
        raise ValueError(
            f"{label} is asymmetric (relative asymmetry {rel:.3e} > "
            f"{_SYM_RTOL_HARD:.1e}). This indicates a structurally wrong "
            "fine-level weighting (D on the wrong side) or an unsupported "
            "mesh/stencil (e.g. a stretched lat-lon grid); refusing to "
            "build the V-cycle. Jacobi preconditioning remains available "
            "via set_preconditioner('jacobi')."
        )
    if rel > _SYM_RTOL_CLEAN:
        warnings.warn(
            f"{label}: relative asymmetry {rel:.3e} is above float64 "
            "roundoff but within the storage-precision tier (typical for "
            "filter caches saved in float32); symmetrizing and proceeding.",
            RuntimeWarning, stacklevel=2)
    return ((A + A.T) * 0.5).tocsr()


def _power_lam_max(A, inv_diag, seed):
    """Largest eigenvalue of diag(A)^-1 A by seeded power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(_POWER_ITERS):
        w = inv_diag * (A @ v)
        lam = np.linalg.norm(w)
        if lam == 0.0:
            return 1.0
        v = w / lam
    return float(lam)


def _to_coo_jnp(A):
    C = A.tocoo()
    return (jnp.asarray(C.data, dtype=jnp.float64),
            jnp.asarray(C.row), jnp.asarray(C.col), int(C.shape[0]))


def setup_vcycle(S, area, k, n, P_ops, *, degree=3, alpha=4.0, n_cycles=1,
                 seed=42, lam_safety=1.1):
    """Build all per-(k, n) V-cycle state on the symmetrized operator D @ A.

    Parameters
    ----------
    S : scipy.sparse matrix
        The PSD-convention stencil ``D^{-1} K`` (K symmetric). Pass ``-S``
        for the lat-lon family, whose stencil is assembled negative.
    area : np.ndarray
        Lumped areas (diagonal of D); must match the weighting baked into S.
    k : float
        Filter wavenumber (scalar only).
    n : int
        Filter order (1 = harmonic, 2 = biharmonic).
    P_ops : list of scipy.sparse.csr_matrix
        Prolongators from :func:`build_hierarchy` (k-independent).
    degree, alpha, n_cycles, seed, lam_safety : optional
        Chebyshev degree per pre/post smooth, spectral interval divisor
        (damps [lam_max/alpha, lam_max]), V-cycles per application, power
        iteration seed, and the safety factor on the lam_max estimate
        (Chebyshev only diverges when lam_max is underestimated by >= 25%;
        the measured worst underestimate is 5%, so 1.1 removes that risk
        class for about one extra CG iteration).

    Returns
    -------
    VcycleData
    """
    area = np.asarray(area, dtype=np.float64)
    A = filter_matrix(S, k, n)
    A_hat = sp.diags(area) @ A
    A_hat = _assert_roundoff_symmetric(A_hat.tocsr(), "A_hat_0 = D*A")

    levels = [A_hat]
    for i, P in enumerate(P_ops):
        A_c = (P.T @ (levels[-1] @ P)).tocsr()
        A_c.eliminate_zeros()
        A_c = _assert_roundoff_symmetric(A_c, f"A_hat_{i + 1} = P^T A_hat P")
        levels.append(A_c)

    inv_diags, lam_max = [], []
    for lvl, A_l in enumerate(levels[:-1]):
        d = A_l.diagonal()
        if not (d > 0.0).all():
            raise ValueError(f"V-cycle level {lvl}: non-positive diagonal in A_hat")
        inv_d = 1.0 / d
        inv_diags.append(jnp.asarray(inv_d, dtype=jnp.float64))
        lam_max.append(float(lam_safety) * _power_lam_max(A_l, inv_d, seed))

    A_coarse = np.asarray(levels[-1].todense(), dtype=np.float64)
    try:
        chol, lower = sla.cho_factor(A_coarse, lower=True)
    except sla.LinAlgError:
        jitter = 1e-12 * np.trace(A_coarse) / A_coarse.shape[0]
        chol, lower = sla.cho_factor(
            A_coarse + jitter * np.eye(A_coarse.shape[0]), lower=True)

    return VcycleData(
        A_coo=tuple(_to_coo_jnp(A_l) for A_l in levels[:-1]),
        P_coo=tuple(_to_coo_jnp(P) for P in P_ops),
        R_coo=tuple(_to_coo_jnp(P.T.tocsr()) for P in P_ops),
        inv_diags=tuple(inv_diags),
        lam_max=tuple(lam_max),
        coarse_chol=jnp.asarray(chol, dtype=jnp.float64),
        coarse_lower=bool(lower),
        sizes=tuple(A_l.shape[0] for A_l in levels),
        degree=int(degree), alpha=float(alpha), n_cycles=int(n_cycles),
    )


def _coo_matvec(coo, x):
    ss, ii, jj, n_rows = coo
    return jnp.zeros(n_rows, dtype=x.dtype).at[ii].add(ss * x[jj])


def _cheb_smooth(A_coo, inv_diag, theta, delta, degree, b, x=None):
    """Chebyshev(degree) on the diag-scaled level operator.

    Damps the interval [theta - delta, theta + delta]. A fixed polynomial in
    the operator, so the whole V-cycle stays linear and symmetric on the SPD
    A_hat; no inner products, matvec-only. The ``x is None`` branch exploits
    the zero initial guess of pre-smoothing, saving one matvec per level.
    """
    sigma = theta / delta
    rho = 1.0 / sigma
    if x is None:
        r = b
        z = inv_diag * r
        d = z / theta
        x = d
    else:
        r = b - _coo_matvec(A_coo, x)
        z = inv_diag * r
        d = z / theta
        x = x + d
    for _ in range(degree - 1):
        rho_new = 1.0 / (2.0 * sigma - rho)
        r = b - _coo_matvec(A_coo, x)
        z = inv_diag * r
        d = (rho_new * rho) * d + (2.0 * rho_new / delta) * z
        x = x + d
        rho = rho_new
    return x


def make_vcycle_preconditioner(data: VcycleData) -> Callable:
    """Return ``M(r_hat) ~= A_hat^{-1} r_hat``: JAX-traceable, linear, SPD.

    The caller must pass residuals of the *symmetrized* system ``(D A) x =
    D b``; inside ``jax.scipy.sparse.linalg.cg`` on that system this is
    automatic. The recursion unrolls at trace time (the level count is
    static Python), so ``M`` works under jit on CPU and GPU identically.
    With an empty hierarchy the cycle degenerates to the exact dense solve.
    """
    n_levels = len(data.A_coo)          # smoothed levels; coarse solve is extra
    intervals = []
    for lam in data.lam_max:
        lam_min = lam / data.alpha
        intervals.append(((lam + lam_min) / 2.0, (lam - lam_min) / 2.0))

    def cycle(level, b):
        if level == n_levels:
            return jsl.cho_solve((data.coarse_chol, data.coarse_lower), b)
        A_coo = data.A_coo[level]
        inv_d = data.inv_diags[level]
        theta, delta = intervals[level]
        x = _cheb_smooth(A_coo, inv_d, theta, delta, data.degree, b)
        r = b - _coo_matvec(A_coo, x)
        e = cycle(level + 1, _coo_matvec(data.R_coo[level], r))
        x = x + _coo_matvec(data.P_coo[level], e)
        return _cheb_smooth(A_coo, inv_d, theta, delta, data.degree, b, x)

    def M(r):
        b_hat = jnp.asarray(r, dtype=jnp.float64)
        x = cycle(0, b_hat)
        for _ in range(data.n_cycles - 1):
            x = x + cycle(0, b_hat - _coo_matvec(data.A_coo[0], x))
        return x

    return M


def pcg_kernel(apply_A, b, M, tol, maxiter, x0=None):
    """Preconditioned CG loop as a pure JAX computation (jit-safe).

    Replicates the algorithm and stopping rule of
    ``jax.scipy.sparse.linalg.cg`` (``||r||_2 <= tol * ||b||_2``). Returns
    ``(x, iterations)`` as JAX values; see :func:`pcg_counted` for the
    convenience wrapper.
    """
    import jax

    b = jnp.asarray(b, dtype=jnp.float64)
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)
    atol2 = (tol * jnp.linalg.norm(b)) ** 2

    r = b - apply_A(x)
    z = M(r)
    gamma = jnp.vdot(r, z)

    def cond(state):
        x, r, p, gamma, it = state
        return (jnp.vdot(r, r) > atol2) & (it < maxiter)

    def body(state):
        x, r, p, gamma, it = state
        q = apply_A(p)
        alpha = gamma / jnp.vdot(p, q)
        x = x + alpha * p
        r = r - alpha * q
        z = M(r)
        gamma_new = jnp.vdot(r, z)
        p = z + (gamma_new / gamma) * p
        return x, r, p, gamma_new, it + 1

    x, r, p, gamma, it = jax.lax.while_loop(cond, body, (x, r, z, gamma, 0))
    return x, it


def pcg_counted(apply_A, b, M, x0=None, *, tol, maxiter):
    """Preconditioned CG that reports iteration count and true residual.

    Counts are representative of the production ``jax.scipy`` CG (same
    algorithm and stopping rule); used by tests and benchmarks only.
    Returns ``(x, iterations, final relative residual)``.
    """
    x, it = pcg_kernel(apply_A, b, M, tol, maxiter, x0)
    b = jnp.asarray(b, dtype=jnp.float64)
    relres = float(jnp.linalg.norm(b - apply_A(x)) / jnp.linalg.norm(b))
    return x, int(it), relres


def validate_scalar_k(kl):
    """Return the scalar filter wavenumber, rejecting spatially varying k.

    The V-cycle is built for one operator per (k, n); a varying k breaks the
    symmetry of D*A (it would need a diagonal similarity transform). Accepts
    a scalar or a constant array.
    """
    if isinstance(kl, (float, int, np.number)):
        return float(kl)
    uniq = np.unique(np.asarray(kl))
    if uniq.size != 1:
        raise ValueError(
            "The V-cycle preconditioner supports only a scalar filter "
            "scale; got a spatially varying k. Use "
            "set_preconditioner('jacobi') for variable scales.")
    return float(uniq[0])


def solve_with_vcycle(*, ss, ii, jj, area, n_size, n, k, apply_A, b_pert,
                      x0_pert, tol, maxiter, options, cache, tag):
    """Solve ``A x = b_pert`` via CG on the symmetrized system with M = V-cycle.

    The CG runs on ``(D A) x = D b_pert`` (SPD), preconditioned by one
    V-cycle per iteration. ``ss/ii/jj`` must be the PSD-convention stencil
    ``S = D^{-1} K`` (pass ``-ss`` for the lat-lon family). ``apply_A`` is
    the host's matrix-free operator for the *original* system; the
    symmetrized operator is ``area * apply_A(x)``. ``cache`` is the filter
    instance's runtime dict; ``tag`` distinguishes the node/element/lat-lon
    systems sharing one instance.

    Because CG stops on the D-weighted residual, the unweighted residual is
    verified afterwards (with one bounded tighter-tolerance retry); jax's cg
    reports no status, so this gate is the convergence check.
    """
    from jax.scipy.sparse.linalg import cg

    from .utils import SolverNotConvergedError

    k = float(k)
    area_np = np.asarray(area, dtype=np.float64)
    # The system fingerprint guards against a re-prepared instance reusing a
    # hierarchy from the previous mesh (same tag, different operator).
    sys_id = (tag, int(n_size), int(len(ss)))
    h_key = ("P",) + sys_id
    if h_key not in cache:
        S = sp.csr_matrix(
            (np.asarray(ss, dtype=np.float64),
             (np.asarray(ii), np.asarray(jj))), shape=(n_size, n_size))
        cache[("S",) + sys_id] = S
        cache[h_key] = build_hierarchy(
            S, area_np,
            max_levels=options["max_levels"], max_coarse=options["max_coarse"],
            strength=options["strength"], seed=options["seed"])
    d_key = ("data",) + sys_id + (int(n), k)
    if d_key not in cache:
        cache[d_key] = setup_vcycle(
            cache[("S",) + sys_id], area_np, k, n, cache[h_key],
            degree=options["degree"], alpha=options["alpha"],
            n_cycles=options["n_cycles"], seed=options["seed"],
            lam_safety=options["lam_safety"])

    M = make_vcycle_preconditioner(cache[d_key])
    area_j = jnp.asarray(area_np)
    apply_A_hat = lambda x: area_j * apply_A(x)
    b_hat = area_j * b_pert

    x, _ = cg(apply_A_hat, b_hat, x0=x0_pert, tol=tol, maxiter=maxiter, M=M)
    b_norm = jnp.linalg.norm(b_pert)
    if float(b_norm) == 0.0:
        return x
    rel = float(jnp.linalg.norm(b_pert - apply_A(x)) / b_norm)
    if rel > tol:
        x, _ = cg(apply_A_hat, b_hat, x0=x, tol=tol * 0.1, maxiter=maxiter, M=M)
        rel = float(jnp.linalg.norm(b_pert - apply_A(x)) / b_norm)
        if rel > tol:
            raise SolverNotConvergedError(
                "V-cycle-preconditioned CG did not reach the requested "
                f"tolerance (achieved {rel:.3e} > {tol:.3e})",
                [f"tag={tag}", f"n={n}", f"k={k}"])
    return x
