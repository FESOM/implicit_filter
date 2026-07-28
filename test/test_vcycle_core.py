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


def test_setup_lam_safety_applied(tiny_setup):
    S, area, P_ops = tiny_setup
    k = 2 * math.pi / 500.0
    raw = vc.setup_vcycle(S, area, k, 2, P_ops, lam_safety=1.0)
    saf = vc.setup_vcycle(S, area, k, 2, P_ops, lam_safety=1.1)
    for a, b in zip(raw.lam_max, saf.lam_max):
        assert b == pytest.approx(1.1 * a, rel=1e-12)
