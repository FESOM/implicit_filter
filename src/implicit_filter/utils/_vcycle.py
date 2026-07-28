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
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import jax.numpy as jnp
import jax.scipy.linalg as jsl

_SYM_RTOL = 1e-10
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

    Roundoff-scale asymmetry (from the D-weighted product) is averaged away;
    anything above ``_SYM_RTOL`` relative means the fine-level weighting is
    wrong (D on the wrong side) or the mesh/stencil is unsupported, and
    silently symmetrizing would hide a broken preconditioner.
    """
    diff = (A - A.T).tocoo()
    denom = max(np.abs(A.data).max(), 1e-300)
    rel = (np.abs(diff.data).max() / denom) if diff.nnz else 0.0
    if rel > _SYM_RTOL:
        raise ValueError(
            f"{label} is asymmetric (relative asymmetry {rel:.3e} > "
            f"{_SYM_RTOL:.1e}). This indicates a structurally wrong "
            "fine-level weighting (D on the wrong side) or an unsupported "
            "mesh/stencil; refusing to build the V-cycle. Jacobi "
            "preconditioning remains available via "
            "set_preconditioner('jacobi')."
        )
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
