"""Integration tests: V-cycle preconditioner through the public filter API.

Comparison bounds: compute() solves the perturbation system ttw = data -
A*data, so its tol=1e-6 is relative to ||ttw||, which at stiff (n, k) is
orders of magnitude larger than ||data||. Jacobi-CG stops there; the
V-cycle typically overshoots to near machine precision (it contracts ~2
digits per iteration). Hence the V-cycle result is checked strictly
against a direct dense solve, and against Jacobi only at the accuracy
level Jacobi actually delivers.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from implicit_filter import LatLonFilter, TriangularFilter
from implicit_filter.utils import _vcycle as vc
from test_vcycle_core import structured_tri_mesh


def _direct_solve(ss, ii, jj, n_size, n, k, rhs):
    """Dense direct solution of (I + 2 (S/k^2)^n) x = rhs from COO triplets."""
    S = sp.csr_matrix((np.asarray(ss, dtype=np.float64),
                       (np.asarray(ii), np.asarray(jj))),
                      shape=(n_size, n_size))
    A = vc.filter_matrix(S, k, n)
    return np.linalg.solve(np.asarray(A.todense()), rhs)


@pytest.fixture(scope="module")
def prepared():
    x, y, tri = structured_tri_mesh(11, 11, 10.0)
    f = TriangularFilter()
    f.prepare(len(x), len(tri), tri, x, y, meshtype="m", cartesian=True,
              filter_elements=True)
    return f


def test_preconditioner_api_roundtrip(prepared):
    assert prepared.get_preconditioner() == "jacobi"             # default untouched
    prepared.set_preconditioner("vcycle")
    assert prepared.get_preconditioner() == "vcycle"
    prepared.set_preconditioner("none")
    assert prepared.get_preconditioner() == "none"
    prepared.set_preconditioner(None)                            # alias for "none"
    assert prepared.get_preconditioner() == "none"
    prepared.set_preconditioner("jacobi")
    with pytest.raises(ValueError, match="Unknown preconditioner"):
        prepared.set_preconditioner("amg")
    with pytest.raises(ValueError, match="Unknown V-cycle option"):
        prepared.set_preconditioner("vcycle", degre=3)


def test_none_preconditioner_matches_jacobi(prepared):
    rng = np.random.default_rng(14)
    data = rng.normal(size=int(prepared._n2d))
    k = 2 * math.pi / 100.0                                      # easy config
    prepared.set_preconditioner("jacobi")
    ref = prepared.compute(1, k, data)
    prepared.set_preconditioner("none")                          # plain CG
    out = prepared.compute(1, k, data)
    prepared.set_preconditioner("jacobi")
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_nodal_vcycle_matches_direct_and_jacobi(prepared):
    rng = np.random.default_rng(11)
    data = rng.normal(size=int(prepared._n2d))
    k = 2 * math.pi / 300.0                                      # stiff: (L/dx)^4 ~ 8e5
    x_true = _direct_solve(prepared._ss, prepared._ii, prepared._jj,
                           int(prepared._n2d), 2, k, data)
    prepared.set_preconditioner("jacobi")
    ref = prepared.compute(2, k, data)
    prepared.set_preconditioner("vcycle")
    out = prepared.compute(2, k, data)
    prepared.set_preconditioner("jacobi")
    # V-cycle: strict agreement with the direct solve.
    assert np.abs(out - x_true).max() < 1e-8
    # Jacobi stops at tol relative to the (large) perturbation RHS; agree
    # at the level it actually delivers.
    assert np.abs(out - ref).max() < 1e-3


def test_element_vcycle_matches_direct_and_jacobi(prepared):
    rng = np.random.default_rng(12)
    data = rng.normal(size=int(prepared._e2d))
    k = 2 * math.pi / 300.0
    x_true = _direct_solve(prepared._ss_e, prepared._ii_e, prepared._jj_e,
                           int(prepared._e2d), 2, k, data)
    prepared.set_preconditioner("jacobi")
    ref = prepared.compute(2, k, data, on="elements")
    prepared.set_preconditioner("vcycle")
    out = prepared.compute(2, k, data, on="elements")
    prepared.set_preconditioner("jacobi")
    assert np.abs(out - x_true).max() < 1e-8
    # The element system is stiffer than the nodal one; Jacobi's loose stop
    # leaves ~1e-2 error here (measured), so this is only a sanity bound.
    assert np.abs(out - ref).max() < 5e-2


def test_varying_k_raises(prepared):
    prepared.set_preconditioner("vcycle")
    kl = np.full(int(prepared._n2d), 2 * math.pi / 300.0)
    kl[0] *= 2.0
    with pytest.raises(ValueError, match="scalar filter scale"):
        prepared.compute(2, kl, np.ones(int(prepared._n2d)))
    prepared.set_preconditioner("jacobi")


def test_constant_k_array_accepted(prepared):
    prepared.set_preconditioner("vcycle")
    kl = np.full(int(prepared._n2d), 2 * math.pi / 300.0)
    out = prepared.compute(1, kl, np.ones(int(prepared._n2d)))
    prepared.set_preconditioner("jacobi")
    assert np.isfinite(out).all()


def test_save_load_unaffected_by_preconditioner(prepared, tmp_path):
    prepared.set_preconditioner("vcycle")
    p = tmp_path / "cache.npz"
    prepared.save_to_file(str(p))
    loaded = TriangularFilter.load_from_file(str(p))
    assert loaded.get_preconditioner() == "jacobi"               # runtime state not persisted
    prepared.set_preconditioner("jacobi")


@pytest.fixture(scope="module")
def prepared_latlon():
    # Uniform cartesian grid, no land mask; 45x40 = 1800 points exceeds
    # max_coarse=1000 so the hierarchy is genuinely multilevel.
    lon = np.linspace(0.0, 20.0, 45)
    lat = np.linspace(-10.0, 10.0, 40)
    f = LatLonFilter()
    f.prepare(lat, lon, cartesian=True, local=True)
    return f


def test_latlon_vcycle_matches_direct_and_jacobi(prepared_latlon):
    f = prepared_latlon
    rng = np.random.default_rng(13)
    data2d = rng.normal(size=(f._nx, f._ny))
    k, n = 0.5, 2                     # (L/dx)^4 ~ 6e5: stiff
    # The lat-lon stencil is assembled negative-semidefinite and the solve
    # scales by -1/k^2, so the PSD-convention operator uses -S.
    x_true = _direct_solve(-np.asarray(f._ss), f._ii, f._jj, int(f._n2d),
                           n, k, np.reshape(np.asarray(data2d), int(f._n2d)))
    f.set_preconditioner("jacobi")
    ref = f.compute(n, k, data2d)
    f.set_preconditioner("vcycle")
    out = f.compute(n, k, data2d)
    # The vcycle path must actually have run (this config is mild enough
    # that Jacobi would also pass the numeric bounds).
    p_keys = [key for key in f.vcycle_cache
              if key[0] == "P" and key[1] == "latlon"]
    assert p_keys, f.vcycle_cache.keys()
    assert len(f.vcycle_cache[p_keys[0]]) >= 1           # genuinely multilevel
    f.set_preconditioner("jacobi")
    assert np.abs(np.reshape(out, int(f._n2d)) - x_true).max() < 1e-8
    assert np.abs(out - ref).max() < 5e-2


def test_latlon_stretched_grid_supported_via_area_squared_weight():
    # Stretched tensor-product grids are asymmetric under the plain area
    # weight (7e-2 relative on this fixture) but exactly symmetric under
    # area^2, so the V-cycle must work -- and match a direct solve.
    lon = np.linspace(0.0, 20.0, 45)
    lat = np.concatenate([np.linspace(-10.0, 0.0, 15),
                          np.linspace(0.5, 10.0, 25)])   # non-uniform spacing
    f = LatLonFilter()
    f.prepare(lat, lon, cartesian=True, local=True)
    rng = np.random.default_rng(17)
    data2d = rng.normal(size=(f._nx, f._ny))
    k, n = 0.5, 2
    x_true = _direct_solve(-np.asarray(f._ss), f._ii, f._jj, int(f._n2d),
                           n, k, np.reshape(np.asarray(data2d), int(f._n2d)))
    f.set_preconditioner("vcycle")
    out = f.compute(n, k, data2d)
    f.set_preconditioner("jacobi")
    assert np.abs(np.reshape(out, int(f._n2d)) - x_true).max() < 1e-8


def test_full_metric_terms_rejected():
    x, y, tri = structured_tri_mesh(11, 11, 10.0)
    f = TriangularFilter()
    f.prepare(len(x), len(tri), tri, x, y, meshtype="r", cartesian=False,
              full=True)
    f.set_preconditioner("vcycle")
    with pytest.raises(NotImplementedError, match="metric-terms"):
        f.compute_velocity(1, 2 * math.pi / 300.0,
                           np.ones(int(f._n2d)), np.ones(int(f._n2d)))
