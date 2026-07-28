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
