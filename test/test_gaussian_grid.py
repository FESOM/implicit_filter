"""
Reduced Gaussian grid definitions: Gaussian latitudes, ECMWF N/O points per
row, storage order, and the spherical Delaunay triangulation.
"""
import numpy as np
import pytest

import implicit_filter
from implicit_filter.utils._gaussian_grid import (
    classical_pl,
    gaussian_latitudes,
    octahedral_pl,
    reduced_gaussian_grid,
    spherical_triangulation,
)
from implicit_filter.utils._reduced_gaussian_tables import CLASSICAL_PL

# numberOfDataPoints of the ecCodes samples reduced_gg_pl_<N>_grib2
CLASSICAL_TOTALS = {
    32: 6114, 48: 13280, 64: 24572, 80: 35718, 96: 50662, 128: 88838,
    160: 138346, 200: 213988, 256: 348528, 320: 542080, 400: 843490,
    512: 1373624, 640: 2140702, 1024: 5447118, 1280: 8505906, 2000: 20696844,
}


class TestGaussianLatitudes:
    def test_n1_is_arcsin_one_over_sqrt3(self):
        np.testing.assert_allclose(
            gaussian_latitudes(1), [35.26438968, -35.26438968], atol=1e-8)

    @pytest.mark.parametrize("N, first", [
        (32, 87.863799), (96, 89.284228), (320, 89.784876)])
    def test_first_latitude_matches_ecmwf(self, N, first):
        # latitudeOfFirstGridPointInDegrees of the ecCodes samples (GRIB stores
        # microdegrees, so the reference itself is only good to 5e-7)
        assert abs(gaussian_latitudes(N)[0] - first) < 2e-6

    def test_count_symmetry_and_order(self):
        lats = gaussian_latitudes(48)
        assert lats.shape == (96,)
        np.testing.assert_allclose(lats, -lats[::-1], atol=1e-13)
        assert np.all(np.diff(lats) < 0)          # north to south
        assert lats[0] < 90.0 and lats[-1] > -90.0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            gaussian_latitudes(0)


@pytest.mark.parametrize("func", [gaussian_latitudes, octahedral_pl, classical_pl])
@pytest.mark.parametrize("bad_n", [2.9, True])
def test_non_integral_or_bool_n_is_rejected(func, bad_n):
    with pytest.raises(TypeError):
        func(bad_n)


class TestPointsPerRow:
    def test_octahedral_formula(self):
        pl = octahedral_pl(96)
        assert pl.shape == (192,)
        assert pl[:3].tolist() == [20, 24, 28]
        assert pl[95] == 4 * 96 + 16
        assert np.array_equal(pl, pl[::-1])
        assert pl.sum() == 4 * 96 * (96 + 9) == 40320

    @pytest.mark.parametrize("N", sorted(CLASSICAL_TOTALS))
    def test_classical_tables_are_consistent(self, N):
        half = CLASSICAL_PL[N]
        assert len(half) == N
        assert all(p > 0 for p in half)
        pl = classical_pl(N)
        assert pl.shape == (2 * N,)
        assert np.array_equal(pl, pl[::-1])
        assert pl.sum() == CLASSICAL_TOTALS[N]

    def test_n320_matches_era5(self):
        pl = classical_pl(320)
        assert pl[:5].tolist() == [18, 25, 36, 40, 45]
        assert pl.max() == 1280
        assert pl.sum() == 542080

    def test_unknown_classical_grid_lists_available(self):
        with pytest.raises(ValueError, match="N100") as exc:
            classical_pl(100)
        assert "320" in str(exc.value)


class TestReducedGaussianGrid:
    def test_o96_layout(self):
        lat, lon = reduced_gaussian_grid("O96")
        assert lat.shape == lon.shape == (40320,)
        assert np.all(np.diff(lat) <= 0)          # rows north to south
        first_row = lat == lat[0]
        assert first_row.sum() == 20
        np.testing.assert_allclose(lon[first_row], np.arange(20) * 18.0)
        assert lon.min() == 0.0 and lon.max() < 360.0
        assert abs(lat[0] - 89.284228) < 2e-6

    def test_n320_size(self):
        lat, lon = reduced_gaussian_grid("N320")
        assert lat.shape == lon.shape == (542080,)
        assert abs(lat[0] - 89.784876) < 2e-6
        assert (lat == lat[0]).sum() == 18

    def test_name_is_case_insensitive_and_tolerates_spaces(self):
        a = reduced_gaussian_grid("o32")
        b = reduced_gaussian_grid(" O32 ")
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    @pytest.mark.parametrize("name", ["X96", "96", "O", "TCo96", "", "N100"])
    def test_bad_names(self, name):
        with pytest.raises(ValueError):
            reduced_gaussian_grid(name)


def assert_closed_manifold(tri, n):
    """Shared assertions for a valid closed-surface triangulation of n points."""
    assert tri.shape == (2 * n - 4, 3)
    assert tri.dtype.kind == "i"
    assert tri.min() == 0 and tri.max() == n - 1
    assert np.unique(tri).size == n                     # every point is a node
    edges = np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)                          # closed surface


class TestSphericalTriangulation:
    @pytest.mark.parametrize("name", ["O16", "O32", "N32", "N320"])
    def test_closed_manifold(self, name):
        # Exercises the fan triangles that close the polar cap on grids with
        # small polar rings (N32's first row has 20 points, N320's has 18,
        # O16's has 20), which sit close to the _HOLE_FACTOR margin in
        # spherical_triangulation. N320 (~540k points, ~7s) is included
        # because it stays comfortably under the ~15s budget for this test.
        lat, lon = reduced_gaussian_grid(name)
        tri = spherical_triangulation(lat, lon)
        assert_closed_manifold(tri, lat.size)

    def test_any_longitude_convention(self):
        lat, lon = reduced_gaussian_grid("O16")
        tri_a = spherical_triangulation(lat, lon)
        tri_b = spherical_triangulation(lat, np.where(lon >= 180.0, lon - 360.0, lon))
        # The two longitude conventions place the points identically on the
        # sphere, but the wrap at +/-180 changes which points are adjacent in
        # storage order, so qhull may triangulate coplanar polar-cap and
        # equatorial-quad regions differently -- tri_a and tri_b need not be
        # identical (or even the same shape in principle). Both must still be
        # valid closed-surface triangulations of the same point set.
        assert_closed_manifold(tri_a, lat.size)
        assert_closed_manifold(tri_b, lat.size)

    def test_rejects_regional_points(self):
        lon, lat = np.meshgrid(np.linspace(0.0, 10.0, 6), np.linspace(0.0, 10.0, 6))
        with pytest.raises(ValueError, match="sphere"):
            spherical_triangulation(lat.ravel(), lon.ravel())

    def test_rejects_missing_polar_cap(self):
        # The origin is still inside the hull (the remaining points still
        # surround it), but the deleted 45N-90N cap leaves a hole the hull
        # has to bridge with oversized triangles -- caught by the hole
        # detector, not the origin-inside-hull check.
        lat, lon = reduced_gaussian_grid("O16")
        keep = lat < 45.0
        with pytest.raises(ValueError, match="span a gap"):
            spherical_triangulation(lat[keep], lon[keep])

    def test_rejects_missing_hemisphere(self):
        # Caught by the origin-inside-hull check (the remaining points no
        # longer surround the origin), not the hole detector.
        lat, lon = reduced_gaussian_grid("O16")
        keep = lat <= 0.0
        with pytest.raises(ValueError, match="sphere"):
            spherical_triangulation(lat[keep], lon[keep])

    def test_rejects_ocean_only_subset(self):
        # A continent-sized longitude band deleted from an otherwise global
        # grid -- the kind of masked/subsetted field a user might pass by
        # mistake. Caught by the hole detector.
        lat, lon = reduced_gaussian_grid("O32")
        keep = (lon < 100.0) | (lon > 160.0)
        with pytest.raises(ValueError, match="span a gap"):
            spherical_triangulation(lat[keep], lon[keep])

    def test_regular_latlon_like_grid_is_rejected_unless_check_coverage_false(self):
        # A dense, exactly co-circular polar ring (144 equally spaced points,
        # none at the pole itself) forces scipy's ConvexHull to close the
        # polar cap with a single-vertex fan whose spoke edges are far longer
        # than the ring spacing -- a real geometric consequence of
        # triangulating a boundary-only convex polygon on a sphere, not a
        # data hole. The hole heuristic cannot distinguish the two, so this
        # complete grid is rejected by default and must opt out.
        lats = np.arange(88.75, -90.0, -2.5)
        lons = np.arange(144) * (360.0 / 144)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        lat, lon = LAT.ravel(), LON.ravel()
        with pytest.raises(ValueError, match="check_coverage"):
            spherical_triangulation(lat, lon)
        tri = spherical_triangulation(lat, lon, check_coverage=False)
        assert_closed_manifold(tri, lat.size)

    def test_check_coverage_false_keeps_exact_guards(self):
        # check_coverage=False only skips the hole heuristic: the exact
        # checks (shape, hull, origin-inside, duplicate points) still run.
        lon, lat = np.meshgrid(np.linspace(0.0, 10.0, 6), np.linspace(0.0, 10.0, 6))
        with pytest.raises(ValueError, match="sphere"):
            spherical_triangulation(lat.ravel(), lon.ravel(), check_coverage=False)

        lat, lon = reduced_gaussian_grid("O16")
        lat = np.append(lat, lat[100])
        lon = np.append(lon, lon[100])
        with pytest.raises(ValueError):
            spherical_triangulation(lat, lon, check_coverage=False)

    def test_accepts_near_coincident_point(self):
        # Documents that near-duplicate points are the caller's
        # responsibility, not treated as a hole: the median-incident-edge
        # local scale (unlike the old nearest-neighbour distance) is not
        # collapsed by one anomalously short edge.
        lat, lon = reduced_gaussian_grid("O32")
        lat = np.append(lat, lat[1000] + 0.01)
        lon = np.append(lon, lon[1000])
        tri = spherical_triangulation(lat, lon)
        assert tri.shape == (2 * lat.size - 4, 3)

    def test_rejects_duplicate_points(self):
        lat, lon = reduced_gaussian_grid("O16")
        lat = np.append(lat, lat[100])
        lon = np.append(lon, lon[100])
        with pytest.raises(ValueError):
            spherical_triangulation(lat, lon)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            spherical_triangulation(np.zeros(5), np.zeros(4))
        with pytest.raises(ValueError):
            spherical_triangulation(np.zeros((2, 2)), np.zeros((2, 2)))


class TestExports:
    @pytest.mark.parametrize("name", [
        "gaussian_latitudes", "reduced_gaussian_grid", "spherical_triangulation"])
    def test_exported_from_package_root(self, name):
        assert hasattr(implicit_filter, name)


class TestAgainstEccodes:
    """Provenance: the bundled tables and latitudes reproduce ecCodes."""

    @pytest.mark.parametrize("N", [32, 320])
    def test_tables(self, N):
        eccodes = pytest.importorskip("eccodes")
        h = eccodes.codes_grib_new_from_samples(f"reduced_gg_pl_{N}_grib2")
        try:
            pl = np.array(eccodes.codes_get_array(h, "pl"), dtype=int)
        finally:
            eccodes.codes_release(h)
        assert np.array_equal(classical_pl(N), pl)

    @pytest.mark.parametrize("N", [32, 320])
    def test_latitudes(self, N):
        eccodes = pytest.importorskip("eccodes")
        ref = eccodes.codes_get_gaussian_latitudes(N)
        ref = np.array([ref[i] for i in range(2 * N)])
        np.testing.assert_allclose(gaussian_latitudes(N), ref, atol=1e-10)
