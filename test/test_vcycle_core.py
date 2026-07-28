"""Unit tests for the V-cycle preconditioner core (utils/_vcycle.py).

The 121-node structured mesh and the extraction of (S, area) from a prepared
TriangularFilter mirror the fixtures the reference implementation was
validated with (neural_preconditioner, tests/test_vcycle.py); thresholds are
carried over from there. The stationary-contraction test is mutation-proven:
sign-flipped, mis-weighted and double-weighted preconditioners all pass
Krylov-level convergence tests and only that test catches them.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp
import jax.numpy as jnp

from implicit_filter import TriangularFilter
from implicit_filter.utils import _vcycle as vc


def structured_tri_mesh(nx, ny, d_km):
    xs, ys = np.meshgrid(np.arange(nx) * d_km, np.arange(ny) * d_km)
    x, y = xs.ravel(), ys.ravel()
    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            tri.append([n0, n0 + 1, n0 + nx])
            tri.append([n0 + 1, n0 + nx + 1, n0 + nx])
    return x, y, np.array(tri)


@pytest.fixture(scope="module")
def tiny_filter():
    x, y, tri = structured_tri_mesh(11, 11, 10.0)   # 121 nodes, 200 elems, 10 km
    f = TriangularFilter()
    f.prepare(len(x), len(tri), tri, x, y, meshtype="m", cartesian=True,
              filter_elements=True)
    return f


def extract_S_area(f):
    n = int(f._n2d)
    S = sp.csr_matrix((np.asarray(f._ss, dtype=np.float64),
                       (np.asarray(f._ii), np.asarray(f._jj))), shape=(n, n))
    return S, np.asarray(f._area, dtype=np.float64)


@pytest.fixture(scope="module")
def tiny_setup(tiny_filter):
    S, area = extract_S_area(tiny_filter)
    P_ops = vc.build_hierarchy(S, area, max_levels=4, max_coarse=10, seed=42)
    return S, area, P_ops


def test_hierarchy_shapes_chain(tiny_setup):
    S, area, P_ops = tiny_setup
    sizes = [S.shape[0]] + [P.shape[1] for P in P_ops]
    assert sizes[0] == 121
    assert all(a > b for a, b in zip(sizes, sizes[1:]))     # strictly coarsens
    rows = S.shape[0]
    for P in P_ops:
        assert P.shape[0] == rows
        rows = P.shape[1]


def test_hierarchy_deterministic(tiny_filter):
    S, area = extract_S_area(tiny_filter)
    h1 = vc.build_hierarchy(S, area, max_levels=4, max_coarse=10, seed=42)
    h2 = vc.build_hierarchy(S, area, max_levels=4, max_coarse=10, seed=42)
    assert len(h1) == len(h2)
    for P1, P2 in zip(h1, h2):
        assert (P1 != P2).nnz == 0                           # bit-identical


def test_hierarchy_preserves_global_rng(tiny_filter):
    S, area = extract_S_area(tiny_filter)
    rng_before = np.random.get_state()
    vc.build_hierarchy(S, area, max_levels=4, max_coarse=10, seed=1)
    after = np.random.get_state()
    assert rng_before[0] == after[0] and (rng_before[1] == after[1]).all()


def test_single_level_when_small(tiny_filter):
    S, area = extract_S_area(tiny_filter)
    P_ops = vc.build_hierarchy(S, area, max_coarse=1000)     # 121 <= 1000
    assert P_ops == []


# ---------------------------------------------------------------- setup


def test_setup_determinism(tiny_setup):
    S, area, P_ops = tiny_setup
    k = 2 * math.pi / 500.0
    d1 = vc.setup_vcycle(S, area, k, 2, P_ops)
    d2 = vc.setup_vcycle(S, area, k, 2, P_ops)
    assert d1.lam_max == d2.lam_max
    assert np.array_equal(np.asarray(d1.coarse_chol), np.asarray(d2.coarse_chol))


def test_setup_rejects_structural_asymmetry(tiny_setup):
    S, area, P_ops = tiny_setup
    S_bad = S.tolil(copy=True)
    S_bad[0, 1] = S_bad[0, 1] + 1.0        # break K = D S symmetry structurally
    with pytest.raises(ValueError, match="asymmet"):
        vc.setup_vcycle(S_bad.tocsr(), area, 2 * math.pi / 500.0, 2, P_ops)


def test_setup_warns_on_storage_roundoff_asymmetry(tiny_filter):
    # Filter caches saved by older versions store the stencil in float32;
    # the resulting ~1e-8 asymmetry must warn and proceed, not fail.
    S, area = extract_S_area(tiny_filter)
    S32 = S.astype(np.float32).astype(np.float64)
    area32 = area.astype(np.float32).astype(np.float64)
    with pytest.warns(RuntimeWarning, match="storage-precision"):
        data = vc.setup_vcycle(S32, area32, 2 * math.pi / 500.0, 2, [])
    assert data.sizes == (121,)


def test_setup_lam_safety_applied(tiny_setup):
    S, area, P_ops = tiny_setup
    k = 2 * math.pi / 500.0
    raw = vc.setup_vcycle(S, area, k, 2, P_ops, lam_safety=1.0)
    saf = vc.setup_vcycle(S, area, k, 2, P_ops, lam_safety=1.1)
    for a, b in zip(raw.lam_max, saf.lam_max):
        assert b == pytest.approx(1.1 * a, rel=1e-12)


# ---------------------------------------------------------------- apply


def _rhs(S, A, seed=42):
    """Perturbation RHS b = x - A x, the production compute() convention."""
    field = np.random.default_rng(seed).normal(size=S.shape[0])
    return np.asarray(field - A @ field)


def _coo_of(S):
    C = S.tocoo()
    return jnp.asarray(C.row), jnp.asarray(C.col), jnp.asarray(C.data)


def _make_apply_A(S, k, n):
    """The production matrix-free operator: I + 2 (S/k^2)^n via scatter-add."""
    ii, jj, ss = _coo_of(S)
    kl2 = 1.0 / k**2

    def apply_A(x):
        y = x
        for _ in range(n):
            y = jnp.zeros_like(x).at[ii].add(ss * kl2 * y[jj])
        return x + 2.0 * y

    return apply_A


@pytest.fixture(scope="module")
def stiff_M(tiny_setup):
    S, area, P_ops = tiny_setup
    n, L = 2, 500.0
    k = 2 * math.pi / L
    A = vc.filter_matrix(S, k, n)
    data = vc.setup_vcycle(S, area, k, n, P_ops)
    return S, area, A, vc.make_vcycle_preconditioner(data)


def test_stationary_vcycle_contracts_monotonically(stiff_M):
    # x_{m+1} = x_m + M(D(b - A x_m)) must contract on its own -- this catches
    # sign/area-weighting errors that Krylov optimality masks (mutation-proven:
    # M(r)=V(r), M(r)=-V(Dr) and M(r)=V(D^2 r) all pass Krylov-level tests).
    S, area, A, M = stiff_M
    b = _rhs(S, A)
    x = np.zeros(S.shape[0])
    norms = [np.linalg.norm(b)]
    for _ in range(6):
        r = b - A @ x
        x = x + np.asarray(M(jnp.asarray(area * r)))
        norms.append(np.linalg.norm(b - A @ x))
    for prev, cur in zip(norms, norms[1:]):
        assert cur < prev, norms
    assert norms[-1] < 1e-2 * norms[0], norms


def test_M_is_spd(stiff_M):
    # CG requires an SPD preconditioner; guards the smoothing order and
    # R = P^T after any refactor. (The FGMRES-based reference suite had no
    # SPD test; the CG host needs one.)
    S, area, A, M = stiff_M
    rng = np.random.default_rng(7)
    for _ in range(5):
        u = rng.normal(size=S.shape[0])
        v = rng.normal(size=S.shape[0])
        Mu = np.asarray(M(jnp.asarray(u)))
        Mv = np.asarray(M(jnp.asarray(v)))
        sym_err = abs(np.dot(Mu, v) - np.dot(u, Mv))
        scale = np.linalg.norm(Mu) * np.linalg.norm(v) + 1e-300
        assert sym_err / scale < 1e-10
        assert np.dot(u, Mu) > 0.0


def test_M_is_linear(stiff_M):
    S, area, A, M = stiff_M
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=S.shape[0]))
    y = jnp.asarray(rng.normal(size=S.shape[0]))
    lhs = np.asarray(M(2.5 * x + y))
    rhs = 2.5 * np.asarray(M(x)) + np.asarray(M(y))
    assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-10


@pytest.mark.parametrize("n,L,bound", [(1, 100.0, 1e-7), (2, 100.0, 1e-7),
                                       (2, 500.0, 1e-6)])
def test_pcg_vcycle_matches_spsolve(tiny_setup, n, L, bound):
    # (2, 500) bound relaxed with condition-number justification: measured
    # dense cond2 ~ 6e5 on this mesh, so cond*tol is the honest floor.
    S, area, P_ops = tiny_setup
    k = 2 * math.pi / L
    A = vc.filter_matrix(S, k, n)
    b = _rhs(S, A)
    data = vc.setup_vcycle(S, area, k, n, P_ops)
    M = vc.make_vcycle_preconditioner(data)
    apply_A = _make_apply_A(S, k, n)
    area_j = jnp.asarray(area)
    x, iters, relres = vc.pcg_counted(lambda v: area_j * apply_A(v),
                                      jnp.asarray(area * b), M,
                                      tol=1e-9, maxiter=200)
    x_ref = np.asarray(sp.linalg.spsolve(sp.csc_matrix(A), b))
    err = np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref)
    assert err < bound
    assert iters <= 40                                   # failure-region regression


def test_vcycle_beats_jacobi_iterations(tiny_setup):
    S, area, P_ops = tiny_setup
    n, L = 2, 500.0
    k = 2 * math.pi / L
    A = vc.filter_matrix(S, k, n)
    b = _rhs(S, A)
    apply_A = _make_apply_A(S, k, n)
    kl2 = 1.0 / k**2
    diag_A = jnp.asarray(1.0 + 2.0 * (np.asarray(S.diagonal()) * kl2) ** n)
    _, it_jac, _ = vc.pcg_counted(apply_A, jnp.asarray(b),
                                  lambda r: r / diag_A,
                                  tol=1e-9, maxiter=100_000)
    data = vc.setup_vcycle(S, area, k, n, P_ops)
    M = vc.make_vcycle_preconditioner(data)
    area_j = jnp.asarray(area)
    _, it_v, _ = vc.pcg_counted(lambda v: area_j * apply_A(v),
                                jnp.asarray(area * b), M,
                                tol=1e-9, maxiter=100_000)
    assert it_v * 5 <= it_jac        # >=5x fewer iterations on the stiff cell


def test_single_level_is_direct_solve(tiny_filter):
    S, area = extract_S_area(tiny_filter)
    n, L = 2, 500.0
    k = 2 * math.pi / L
    A = vc.filter_matrix(S, k, n)
    b = _rhs(S, A)
    data = vc.setup_vcycle(S, area, k, n, [])         # 121 nodes, no coarsening
    M = vc.make_vcycle_preconditioner(data)
    x = np.asarray(M(jnp.asarray(area * b)))          # M == A_hat^{-1} exactly
    x_ref = np.asarray(sp.linalg.spsolve(sp.csc_matrix(A), b))
    err = np.linalg.norm(x - x_ref)
    assert err / np.linalg.norm(x_ref) < 1e-10
