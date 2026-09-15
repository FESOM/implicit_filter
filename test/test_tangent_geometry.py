"""
meshtype='s': per-triangle geometry evaluated in the tangent plane at the
triangle centroid (gnomonic projection). Second-order accurate in the triangle
size, like the planar branches, but free of their degeneracies -- notably on
the polar caps, where the lon/lat projection of meshtype='r' collapses.
"""
import math

import numpy as np
import pytest

from implicit_filter import (
    TriangularFilter,
    reduced_gaussian_grid,
    spherical_triangulation,
)
from implicit_filter.utils._auxiliary import (
    R_EARTH,
    areas,
    make_tri,
    neighboring_triangles,
    tangent_plane_geometry,
)


def lonlat_patch(n=6, extent=0.5, lat0=0.0, lon0=0.0):
    """Small triangulated lon/lat patch (degrees): lon, lat, tri."""
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    step = extent / (n - 1)
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    for i in range(n):
        lat[i, :] = lat0 + np.arange(n) * step
        lon[:, i] = lon0 + np.arange(n) * step
    return lon.flatten(), lat.flatten(), make_tri(nodnum, n, n)


def geometry(meshtype, lon, lat, tri):
    n2d, e2d = len(lon), len(tri)
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    return areas(n2d, e2d, tri, lon, lat, ne_num, ne_pos, meshtype, False,
                 2 * math.pi, np.ones(e2d, dtype=bool))


def sphere_mesh(name="O16"):
    lat, lon = reduced_gaussian_grid(name)
    return lon, lat, spherical_triangulation(lat, lon)


class TestAgainstPlanarProjection:
    """Far from the poles the two geometries agree to discretisation order."""

    def test_equatorial_patch(self):
        lon, lat, tri = lonlat_patch(lat0=0.0)
        area_r, ea_r, dx_r, dy_r, Mt_r = geometry("r", lon, lat, tri)
        area_s, ea_s, dx_s, dy_s, Mt_s = geometry("s", lon, lat, tri)
        np.testing.assert_allclose(ea_s, ea_r, rtol=1e-4)
        np.testing.assert_allclose(area_s, area_r, rtol=1e-4)
        scale = np.abs(dx_r).max()
        np.testing.assert_allclose(dx_s, dx_r, rtol=1e-4, atol=1e-4 * scale)
        np.testing.assert_allclose(dy_s, dy_r, rtol=1e-4, atol=1e-4 * scale)
        np.testing.assert_allclose(Mt_s, Mt_r, rtol=1e-4, atol=1e-12)

    @pytest.mark.parametrize("lat0", [45.0, -45.0])
    def test_mid_latitude_patch(self, lat0):
        # 'r' uses one cos(lat) per triangle; across a 0.1 degree triangle at
        # 45 degrees that factor varies by ~2e-3, which bounds the
        # disagreement. Both hemispheres, so a sign error in the north/east
        # basis or in Mt cannot hide.
        lon, lat, tri = lonlat_patch(lat0=lat0, lon0=100.0)
        _, ea_r, dx_r, dy_r, Mt_r = geometry("r", lon, lat, tri)
        _, ea_s, dx_s, dy_s, Mt_s = geometry("s", lon, lat, tri)
        np.testing.assert_allclose(ea_s, ea_r, rtol=1e-2)
        scale = np.abs(dx_r).max()
        np.testing.assert_allclose(dx_s, dx_r, rtol=1e-2, atol=1e-2 * scale)
        np.testing.assert_allclose(dy_s, dy_r, rtol=1e-2, atol=1e-2 * scale)
        np.testing.assert_allclose(Mt_s, Mt_r, rtol=1e-4)

    def test_mask_zeroes_element_areas(self):
        lon, lat, tri = lonlat_patch()
        mask = np.ones(len(tri))
        mask[::2] = 0.0
        ea, dx, dy, Mt = tangent_plane_geometry(tri, lon, lat, mask)
        assert np.all(ea[::2] == 0.0)
        assert np.all(ea[1::2] > 0.0)
        assert np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))


class TestSphere:
    def test_total_area_is_the_sphere(self):
        lon, lat, tri = sphere_mesh()
        area, elem_area, dx, dy, Mt = geometry("s", lon, lat, tri)
        assert abs(elem_area.sum() / (4 * math.pi * R_EARTH ** 2) - 1.0) < 5e-3
        np.testing.assert_allclose(area.sum(), elem_area.sum(), rtol=1e-9)
        # O16 node areas are ~1e5 km^2; the 1e-5 clamp the shared tail applies
        # to empty nodes would fail this.
        assert area.min() > 100.0

    def test_polar_fan_triangles_are_regular(self):
        lon, lat, tri = sphere_mesh()
        fan = np.all(np.abs(np.abs(lat[tri]) - lat.max()) < 1e-9, axis=1)
        assert fan.sum() > 0, "the hull closes each polar cap with a fan of triangles"
        ea, dx, dy, Mt = tangent_plane_geometry(tri, lon, lat, np.ones(len(tri)))
        assert np.all(np.isfinite(dx)) and np.all(np.isfinite(dy)) and np.all(np.isfinite(Mt))
        assert ea.min() > 0.0
        # the lon/lat projection degenerates on exactly these triangles
        with np.errstate(divide="ignore", invalid="ignore"):
            _, _, dx_r, _, _ = geometry("r", lon, lat, tri)
        assert not np.all(np.isfinite(dx_r[fan]))

    def test_longitude_shift_invariance(self):
        lon, lat, tri = sphere_mesh()
        ones = np.ones(len(tri))
        ea0, dx0, dy0, Mt0 = tangent_plane_geometry(tri, lon, lat, ones)
        ea1, dx1, dy1, Mt1 = tangent_plane_geometry(
            tri, (lon + 123.456) % 360.0, lat, ones)
        scale = np.abs(dx0).max()
        np.testing.assert_allclose(ea1, ea0, rtol=1e-9)
        np.testing.assert_allclose(dx1, dx0, rtol=1e-8, atol=1e-8 * scale)
        np.testing.assert_allclose(dy1, dy0, rtol=1e-8, atol=1e-8 * scale)
        np.testing.assert_allclose(Mt1, Mt0, rtol=1e-9, atol=1e-15)

    def test_centroid_exactly_at_pole(self):
        # Three points at one latitude, evenly spaced in longitude: the
        # centroid direction is exactly the pole, where the local east vector
        # is undefined and the fallback basis (1, 0, 0), (0, 1, 0) is used.
        lat = np.full(3, 80.0)
        lon = np.array([0.0, 120.0, 240.0])
        tri = np.array([[0, 1, 2]])
        ea, dx, dy, Mt = tangent_plane_geometry(tri, lon, lat, np.ones(1))
        assert np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))
        assert np.all(np.isfinite(Mt))
        assert ea[0] > 0.0
        # P1 basis functions sum to one, so their derivatives sum to zero.
        np.testing.assert_allclose(dx.sum(axis=1), 0.0,
                                   atol=1e-12 * np.abs(dx).max())
        np.testing.assert_allclose(dy.sum(axis=1), 0.0,
                                   atol=1e-12 * np.abs(dy).max())
        # Gnomonic area from the pole basis, derived independently here.
        rlat, rlon = np.radians(lat), np.radians(lon)
        x = R_EARTH * np.cos(rlat) * np.cos(rlon) / np.sin(rlat)
        y = R_EARTH * np.cos(rlat) * np.sin(rlon) / np.sin(rlat)
        expected = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0])
                             - (y[1] - y[0]) * (x[2] - x[0]))
        np.testing.assert_allclose(ea[0], expected, rtol=1e-12)

    def test_metric_factor_is_tan_lat_over_r(self):
        lon, lat, tri = lonlat_patch(lat0=30.0)
        _, _, _, Mt = tangent_plane_geometry(tri, lon, lat, np.ones(len(tri)))
        # centroid latitude vs mean vertex latitude differ at O(h^2)
        expected = np.tan(np.radians(lat[tri].mean(axis=1))) / R_EARTH
        np.testing.assert_allclose(Mt, expected, rtol=1e-4)


class TestPrepare:
    def test_unknown_meshtype(self):
        lon, lat, tri = lonlat_patch()
        with pytest.raises(ValueError, match="meshtype"):
            TriangularFilter().prepare(len(lon), len(tri), tri, lon, lat, meshtype="x")

    def test_s_rejects_cartesian(self):
        lon, lat, tri = lonlat_patch()
        with pytest.raises(ValueError, match="cartesian"):
            TriangularFilter().prepare(
                len(lon), len(tri), tri, lon, lat, meshtype="s", cartesian=True)

    def test_s_rejects_filter_elements(self):
        lon, lat, tri = lonlat_patch()
        with pytest.raises(NotImplementedError, match="filter_elements"):
            TriangularFilter().prepare(
                len(lon), len(tri), tri, lon, lat, meshtype="s", cartesian=False,
                filter_elements=True)

    def test_s_prepares_a_global_mesh_and_preserves_constants(self):
        lon, lat, tri = sphere_mesh()
        filt = TriangularFilter()
        filt.prepare(len(lat), len(tri), tri, lon, lat, meshtype="s", cartesian=False)
        assert np.all(np.isfinite(np.asarray(filt._ss)))
        out = filt.compute(1, 2 * math.pi / 3000.0, np.full(len(lat), 2.5))
        np.testing.assert_allclose(out, 2.5, atol=1e-10)

    def test_s_with_full_metric_terms_matches_r_on_patch(self):
        # Exercises the Mt returned by the 's' branch inside make_smooth on a
        # mid-latitude patch (the coupled system is ill-conditioned near the
        # poles, where Mt = tan(lat)/R blows up). Away from the poles the two
        # geometries must give the same filtered velocity, up to the
        # O(h*tan(lat)) projection error of 'r'.
        #
        # 60 degrees rather than 45: Mt = tan(lat)/R grows with latitude, so
        # the pin bites harder there. Negating the 's' branch's Mt moves the
        # result to 2.22x the tolerance at 60 degrees but only 1.14x at 45,
        # while the honest disagreement stays at 0.08 of it.
        lon, lat, tri = lonlat_patch(n=8, extent=2.0, lat0=60.0, lon0=10.0)
        rng = np.random.default_rng(0)
        u0 = rng.standard_normal(len(lat))
        v0 = rng.standard_normal(len(lat))

        def filtered(meshtype):
            filt = TriangularFilter()
            filt.prepare(len(lat), len(tri), tri, lon, lat, meshtype=meshtype,
                         cartesian=False, full=True)
            return filt.compute_velocity(1, 2 * math.pi / 300.0, u0, v0)

        u_s, v_s = filtered("s")
        u_r, v_r = filtered("r")
        assert u_s.shape == v_s.shape == (len(lat),)
        assert np.all(np.isfinite(u_s)) and np.all(np.isfinite(v_s))
        scale = max(np.abs(u_r).max(), np.abs(v_r).max())
        np.testing.assert_allclose(u_s, u_r, rtol=1e-2, atol=1e-2 * scale)
        np.testing.assert_allclose(v_s, v_r, rtol=1e-2, atol=1e-2 * scale)
