"""
Bit-identity of the vectorised mesh functions in ``_fast_mesh`` against the
pure-Python references in ``_auxiliary``: same neighbour ordering, same
floating-point operation order, same prepare() and compute() output.
"""
import math

import numpy as np
import pytest

import implicit_filter.triangular_filter as tf_module
from implicit_filter import TriangularFilter, reduced_gaussian_grid, spherical_triangulation
from implicit_filter.utils import _auxiliary as aux
from implicit_filter.utils._auxiliary import make_tri
from implicit_filter.utils._fast_mesh import (
    _NUMPY_PAIRWISE_BLOCK,
    fast_areas,
    fast_neighboring_triangles,
    fast_neighbouring_nodes,
    pairwise_row_sums,
)


def lonlat_patch(n=10, extent=4.0, lat0=30.0, lon0=0.0):
    """Triangulated lon/lat patch (degrees) in make_tri layout: lon, lat, tri."""
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    step = extent / (n - 1)
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    for i in range(n):
        lat[i, :] = lat0 + np.arange(n) * step
        lon[:, i] = lon0 + np.arange(n) * step
    return lon.flatten(), lat.flatten(), make_tri(nodnum, n, n)


def fan_mesh(valence=12):
    """Centre node 0 surrounded by `valence` nodes on a circle: node 0 has
    8 <= valence < 16 adjacent triangles, exercising the 8-accumulator path
    of NumPy's summation without a full second block."""
    ang = np.linspace(0.0, 2 * math.pi, valence, endpoint=False)
    x = np.concatenate(([0.0], 100.0 * np.cos(ang) * (1 + 0.1 * np.arange(valence))))
    y = np.concatenate(([0.0], 100.0 * np.sin(ang) * (1 + 0.05 * np.arange(valence))))
    tri = np.array([[0, 1 + i, 1 + (i + 1) % valence] for i in range(valence)])
    return x, y, tri


def hull_mesh(name):
    lat, lon = reduced_gaussian_grid(name)
    return lon, lat, spherical_triangulation(lat, lon)


# --- topology test cases: (label, n2d, tri) ---------------------------------
def topology_cases():
    cases = []
    lon, lat, tri = lonlat_patch()
    cases.append(("patch", lon.size, tri))
    x, y, tri = fan_mesh(12)
    cases.append(("fan12", x.size, tri))
    x, y, tri = fan_mesh(21)
    cases.append(("fan21", x.size, tri))
    # unsigned connectivity: the reference indexes with it happily, so must we
    cases.append(("fan21-uint64", x.size, tri.astype(np.uint64)))
    cases.append(("fan21-uint32", x.size, tri.astype(np.uint32)))
    for name in ("O16", "O32"):
        lon, lat, tri = hull_mesh(name)
        cases.append((name, lon.size, tri))
    # a triangle that repeats a node index, plus an isolated node (index 6)
    cases.append(("degenerate", 7, np.array([[0, 1, 2], [3, 3, 4], [1, 2, 5], [2, 5, 4]])))
    # random triangle soups: repeated indices, duplicate triangles, isolated nodes
    for seed in range(5):
        r = np.random.default_rng(seed)
        cases.append((f"soup{seed}", 30, r.integers(0, 30, size=(60, 3))))
    return cases


TOPOLOGY_CASES = topology_cases()
TOPOLOGY_IDS = [case[0] for case in TOPOLOGY_CASES]


@pytest.mark.parametrize("label, n2d, tri", TOPOLOGY_CASES, ids=TOPOLOGY_IDS)
def test_neighboring_triangles_is_bit_identical(label, n2d, tri):
    e2d = len(tri)
    ref_num, ref_pos = aux.neighboring_triangles(n2d, e2d, tri)
    num, pos = fast_neighboring_triangles(n2d, e2d, tri)
    assert num.dtype == ref_num.dtype and pos.dtype == ref_pos.dtype
    assert pos.shape == ref_pos.shape
    np.testing.assert_array_equal(num, ref_num)
    np.testing.assert_array_equal(pos, ref_pos)


@pytest.mark.parametrize("label, n2d, tri", TOPOLOGY_CASES, ids=TOPOLOGY_IDS)
def test_neighbouring_nodes_is_bit_identical(label, n2d, tri):
    e2d = len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    ref_num, ref_pos = aux.neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    num, pos = fast_neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    assert num.dtype == ref_num.dtype and pos.dtype == ref_pos.dtype
    assert pos.shape == ref_pos.shape
    np.testing.assert_array_equal(num, ref_num)
    np.testing.assert_array_equal(pos, ref_pos)


# --- summation order --------------------------------------------------------
class TestPairwiseRowSums:
    def test_matches_numpy_sum_for_every_length(self):
        rng = np.random.default_rng(1)
        M = _NUMPY_PAIRWISE_BLOCK
        counts = np.repeat(np.arange(0, M + 1), 6)           # every length 0..128, 6 rows each
        rows = len(counts)
        magnitudes = rng.choice([1e-3, 1.0, 1e5], size=(rows, M))
        values = rng.random((rows, M)) * magnitudes
        values[np.arange(M)[None, :] >= counts[:, None]] = 0.0
        expected = np.array([np.sum(values[r, :counts[r]]) for r in range(rows)])
        got = pairwise_row_sums(values, counts)
        assert got.dtype == np.float64 and got.shape == (rows,)
        np.testing.assert_array_equal(got, expected)

    def test_order_matters(self):
        """The test above can discriminate: a naive sequential sum differs."""
        rng = np.random.default_rng(2)
        values = rng.random((500, 21)) * rng.choice([1e-3, 1.0, 1e5], size=(500, 21))
        counts = np.full(500, 21)
        naive = np.zeros(500)
        for i in range(21):
            naive = naive + values[:, i]
        exact = np.array([np.sum(values[r]) for r in range(500)])
        assert np.any(naive != exact)
        np.testing.assert_array_equal(pairwise_row_sums(values, counts), exact)

    def test_zero_count_rows_sum_to_zero(self):
        values = np.ones((3, 5))
        np.testing.assert_array_equal(pairwise_row_sums(values, np.array([0, 0, 0])), 0.0)

    def test_rejects_rows_beyond_numpy_block_size(self):
        with pytest.raises(NotImplementedError):
            pairwise_row_sums(np.zeros((1, 129)), np.array([129]))


# --- geometry: (label, xcoord, ycoord, tri, meshtype, cartesian, cyclic_length)
TWO_PI = 2 * math.pi


def geometry_cases():
    cases = []
    lon, lat, tri = lonlat_patch()
    for meshtype, cartesian in (("m", True), ("r", True), ("r", False)):
        cases.append((f"patch-{meshtype}-{cartesian}", lon, lat, tri, meshtype,
                      cartesian, TWO_PI))
    # date-line crossing patch in [0, 360): cyclic corrections fire
    lon_dl = (lon + 358.0) % 360.0
    cases.append(("dateline-r", lon_dl, lat, tri, "r", False, TWO_PI))
    # float32 coordinates: the reference runs the whole per-triangle chain in
    # single precision, so the fast path must not silently upcast
    lon32, lat32 = lon.astype(np.float32), lat.astype(np.float32)
    cases.append(("patch32-m", lon32, lat32, tri, "m", True, TWO_PI))
    cases.append(("patch32-r", lon32, lat32, tri, "r", False, TWO_PI))
    # a NumPy-typed cyclic_length is strongly typed, so the cyclic correction
    # would promote every element to float64 in a vectorised np.where
    cases.append(("patch32-r-npcyclic", lon32, lat32, tri, "r", False,
                  np.deg2rad(360.0)))
    # the same, with float64 coordinates: this one must stay on the fast path
    cases.append(("patch-r-npcyclic", lon, lat, tri, "r", False, np.deg2rad(360.0)))
    x, y, tri = fan_mesh(12)
    cases.append(("fan12-m", x, y, tri, "m", True, TWO_PI))
    x, y, tri = fan_mesh(21)
    cases.append(("fan21-m", x, y, tri, "m", True, TWO_PI))
    for name in ("O16", "O32"):
        lon, lat, tri = hull_mesh(name)
        cases.append((f"{name}-s", lon, lat, tri, "s", False, TWO_PI))
    return cases


GEOMETRY_CASES = geometry_cases()
GEOMETRY_IDS = [case[0] for case in GEOMETRY_CASES]


@pytest.mark.parametrize("label, xcoord, ycoord, tri, meshtype, cartesian, cyclic_length",
                         GEOMETRY_CASES, ids=GEOMETRY_IDS)
@pytest.mark.parametrize("masked", [False, True])
def test_areas_is_bit_identical(label, xcoord, ycoord, tri, meshtype, cartesian,
                                cyclic_length, masked):
    n2d, e2d = len(xcoord), len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    mask = np.ones(e2d, dtype=bool)
    if masked:
        mask[np.random.default_rng(4).random(e2d) < 0.3] = False
    args = (n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, cartesian,
            cyclic_length, mask)
    ref = aux.areas(*args)
    got = fast_areas(*args)
    for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"), got, ref):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_areas_with_valence_above_the_numpy_block_is_bit_identical():
    """A regular lat/lon point set closes its polar caps with a dense fan, so a
    few nodes have more adjacent triangles than NumPy sums in one block. Those
    nodes take the np.sum fallback in fast_areas and must still be exact."""
    lats = np.arange(88.75, -88.76, -2.5)
    lons = np.arange(0.0, 360.0, 2.5)
    lat = np.repeat(lats, lons.size)
    lon = np.tile(lons, lats.size)
    tri = spherical_triangulation(lat, lon, check_coverage=False)
    n2d, e2d = lat.size, len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    assert ne_num.max() > _NUMPY_PAIRWISE_BLOCK, "case must exercise the fallback"
    args = (n2d, e2d, tri, lon, lat, ne_num, ne_pos, "s", False, 2 * math.pi,
            np.ones(e2d, dtype=bool))
    for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"),
                          fast_areas(*args), aux.areas(*args)):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_areas_with_integer_mask_matches_reference():
    """An integer mask is a weak Python int per element in the reference but a
    strongly typed array here, so with float32 coordinates the two promote
    differently; the value 3 makes that visible (1, 0 and 2 scale exactly)."""
    lon, lat, tri = lonlat_patch()
    lon, lat = lon.astype(np.float32), lat.astype(np.float32)
    n2d, e2d = len(lon), len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    mask = [1, 0, 3] * (e2d // 3) + [1] * (e2d % 3)
    for meshtype, cartesian in (("m", True), ("r", False)):
        args = (n2d, e2d, tri, lon, lat, ne_num, ne_pos, meshtype, cartesian,
                2 * math.pi, mask)
        for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"),
                              fast_areas(*args), aux.areas(*args)):
            assert a.shape == b.shape and a.dtype == b.dtype, (meshtype, name)
            np.testing.assert_array_equal(a, b, err_msg=f"{meshtype} {name}")


def test_fast_areas_delegates_only_in_the_promotion_corner_cases(monkeypatch):
    """The reference loop is the fallback for the promotion corners only; every
    float64 mesh, and every float32 mesh with a plain-float cyclic_length and a
    bool mask, must stay on the vectorised path."""
    lon, lat, tri = lonlat_patch()
    lon32, lat32 = lon.astype(np.float32), lat.astype(np.float32)
    n2d, e2d = len(lon), len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    bool_mask = np.ones(e2d, dtype=bool)
    int_mask = [1, 0, 3] * (e2d // 3) + [1] * (e2d % 3)

    calls = []
    real_areas = aux.areas

    def counting_areas(*args, **kwargs):
        calls.append(args[7])                            # the meshtype
        return real_areas(*args, **kwargs)

    monkeypatch.setattr(aux, "areas", counting_areas)

    def delegates(xc, yc, cyclic_length, mask, meshtype="r", cartesian=False):
        del calls[:]
        fast_areas(n2d, e2d, tri, xc, yc, ne_num, ne_pos, meshtype, cartesian,
                   cyclic_length, mask)
        return bool(calls)

    # delegated: float32 coordinates in the two promotion corners
    assert delegates(lon32, lat32, np.deg2rad(360.0), bool_mask)
    assert delegates(lon32, lat32, 2 * math.pi, int_mask)
    assert delegates(lon32, lat32, 2 * math.pi, int_mask, meshtype="m", cartesian=True)
    # not delegated: float64 coordinates make the corners moot ...
    assert not delegates(lon, lat, np.deg2rad(360.0), bool_mask)
    assert not delegates(lon, lat, 2 * math.pi, int_mask)
    # ... and float32 with a plain-float cyclic_length and a bool mask is exact
    assert not delegates(lon32, lat32, 2 * math.pi, bool_mask)
    assert not delegates(lon32, lat32, np.float32(2 * math.pi).item(), bool_mask)
    # 's' never delegates: tangent_plane_geometry casts to float64 either way
    assert not delegates(lon32, lat32, np.deg2rad(360.0), bool_mask, meshtype="s")


def test_areas_ignores_rows_beyond_e2d():
    """The reference only ever reads tri[n] and mask[n] for n < e2d, so a
    connectivity and mask that run past e2d must not change the answer."""
    lon, lat, tri = lonlat_patch()
    tri = np.vstack([tri, tri[:1]])                      # one row beyond e2d
    n2d, e2d = len(lon), len(tri) - 1
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    mask = np.ones(len(tri), dtype=bool)
    mask[::5] = False
    args = (n2d, e2d, tri, lon, lat, ne_num, ne_pos, "m", True, 2 * math.pi, mask)
    for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"),
                          fast_areas(*args), aux.areas(*args)):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_areas_unknown_meshtype_mirrors_reference():
    lon, lat, tri = lonlat_patch()
    n2d, e2d = len(lon), len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    args = (n2d, e2d, tri, lon, lat, ne_num, ne_pos, "x", False, 2 * math.pi, np.ones(e2d))
    for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"),
                          fast_areas(*args), aux.areas(*args)):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_areas_reads_r_earth_at_call_time(monkeypatch):
    lon, lat, tri = lonlat_patch()
    n2d, e2d = len(lon), len(tri)
    ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
    args = (n2d, e2d, tri, lon, lat, ne_num, ne_pos, "r", False, 2 * math.pi, np.ones(e2d))
    base = fast_areas(*args)[1]
    monkeypatch.setattr(aux, "R_EARTH", 2.0 * aux.R_EARTH)
    got = fast_areas(*args)
    np.testing.assert_allclose(got[1], 4.0 * base, rtol=1e-12)
    # and the patched constant must still give the reference bit for bit
    for name, a, b in zip(("area", "elem_area", "dx", "dy", "Mt"),
                          got, aux.areas(*args)):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


# --- end to end ---------------------------------------------------------------
STATE = ("_ss", "_ii", "_jj", "_area", "_elem_area", "_dx", "_dy", "_mask_n",
         "_ne_num", "_ne_pos", "_en_pos")


def prepare_pair(monkeypatch, n2d, tri, xcoord, ycoord, **kwargs):
    fast = TriangularFilter()
    fast.prepare(n2d, len(tri), tri, xcoord, ycoord, **kwargs)
    monkeypatch.setattr(tf_module, "fast_neighboring_triangles", aux.neighboring_triangles)
    monkeypatch.setattr(tf_module, "fast_neighbouring_nodes", aux.neighbouring_nodes)
    monkeypatch.setattr(tf_module, "fast_areas", aux.areas)
    ref = TriangularFilter()
    ref.prepare(n2d, len(tri), tri, xcoord, ycoord, **kwargs)
    return fast, ref


def assert_same_state(fast, ref, elements=False):
    names = STATE + (("_ss_e", "_ii_e", "_jj_e") if elements else ())
    for name in names:
        a, b = np.asarray(getattr(fast, name)), np.asarray(getattr(ref, name))
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("full", [False, True])
def test_prepare_and_compute_on_lonlat_patch(monkeypatch, full):
    lon, lat, tri = lonlat_patch()
    fast, ref = prepare_pair(monkeypatch, lon.size, tri, lon, lat,
                             meshtype="r", cartesian=False, full=full)
    assert_same_state(fast, ref)
    rng = np.random.default_rng(5)
    data = rng.standard_normal(lon.size)
    if not full:
        # The coupled metric-terms system solves for (u, v) jointly, so the
        # scalar compute() is not applicable to full=True (it raises on the
        # reference path too); compute_velocity() below covers that case.
        np.testing.assert_array_equal(fast.compute(1, 2 * math.pi / 300.0, data),
                                      ref.compute(1, 2 * math.pi / 300.0, data))
    u, v = rng.standard_normal((2, lon.size))
    fu, fv = fast.compute_velocity(1, 2 * math.pi / 300.0, u, v)
    ru, rv = ref.compute_velocity(1, 2 * math.pi / 300.0, u, v)
    np.testing.assert_array_equal(fu, ru)
    np.testing.assert_array_equal(fv, rv)


def test_prepare_and_compute_on_sphere(monkeypatch):
    lon, lat, tri = hull_mesh("O16")
    fast, ref = prepare_pair(monkeypatch, lon.size, tri, lon, lat,
                             meshtype="s", cartesian=False)
    assert_same_state(fast, ref)
    rng = np.random.default_rng(6)
    data = rng.standard_normal(lon.size)
    np.testing.assert_array_equal(fast.compute(1, 2 * math.pi / 3000.0, data),
                                  ref.compute(1, 2 * math.pi / 3000.0, data))


def test_prepare_and_compute_float32_coordinates(monkeypatch):
    """float32 node coordinates (as FesomFilter/IconFilter forward them from
    file) must prepare and filter bit-identically to the reference."""
    lon, lat, tri = lonlat_patch()
    lon, lat = lon.astype(np.float32), lat.astype(np.float32)
    fast, ref = prepare_pair(monkeypatch, lon.size, tri, lon, lat,
                             meshtype="r", cartesian=False)
    assert_same_state(fast, ref)
    rng = np.random.default_rng(8)
    data = rng.standard_normal(lon.size)
    np.testing.assert_array_equal(fast.compute(1, 2 * math.pi / 300.0, data),
                                  ref.compute(1, 2 * math.pi / 300.0, data))


def test_prepare_and_compute_float32_numpy_cyclic_length(monkeypatch):
    """float32 coordinates with a NumPy-typed cyclic_length, as a caller using
    np.deg2rad would pass it, all the way through prepare and compute."""
    lon, lat, tri = lonlat_patch()
    lon, lat = lon.astype(np.float32), lat.astype(np.float32)
    fast, ref = prepare_pair(monkeypatch, lon.size, tri, lon, lat,
                             meshtype="r", cartesian=False,
                             cyclic_length=np.deg2rad(360.0))
    assert_same_state(fast, ref)
    rng = np.random.default_rng(9)
    data = rng.standard_normal(lon.size)
    np.testing.assert_array_equal(fast.compute(1, 2 * math.pi / 300.0, data),
                                  ref.compute(1, 2 * math.pi / 300.0, data))


def test_prepare_with_elements_and_mask(monkeypatch):
    lon, lat, tri = lonlat_patch()
    mask = np.ones(len(tri), dtype=bool)
    mask[::7] = False
    fast, ref = prepare_pair(monkeypatch, lon.size, tri, lon, lat,
                             meshtype="m", cartesian=True, filter_elements=True, mask=mask)
    assert_same_state(fast, ref, elements=True)
    rng = np.random.default_rng(7)
    data = rng.standard_normal(len(tri))
    np.testing.assert_array_equal(fast.compute(1, 2.0, data), ref.compute(1, 2.0, data))
