"""
ReducedGaussianFilter: ECMWF reduced Gaussian grids (N320, O96, ...) filtered
as global spherical triangular meshes. Correctness is pinned against the
analytic transfer function of the implicit filter for spherical harmonics.
"""
import math

import numpy as np
import pytest

import implicit_filter
from implicit_filter import ReducedGaussianFilter, reduced_gaussian_grid
from implicit_filter.utils._auxiliary import R_EARTH

SCALE_KM = 6000.0
K = 2 * math.pi / SCALE_KM


def real_harmonic(l, m, lat_deg, lon_deg):
    sps = pytest.importorskip("scipy.special")
    theta = np.radians(90.0 - lat_deg)
    phi = np.radians(lon_deg)
    if hasattr(sps, "sph_harm_y"):
        y = sps.sph_harm_y(l, m, theta, phi)
    else:  # SciPy < 1.15
        y = sps.sph_harm(m, l, phi, theta)
    return np.real(np.asarray(y))


def analytic_attenuation(l, k=K, n=1):
    """Eigenvalue of the implicit filter for a harmonic of degree l."""
    return 1.0 / (1.0 + 2.0 * ((l * (l + 1) / R_EARTH ** 2) / k ** 2) ** n)


@pytest.fixture(scope="module")
def o32():
    filt = ReducedGaussianFilter()
    filt.prepare_from_grid("O32")
    return filt


@pytest.fixture(scope="module")
def o16():
    filt = ReducedGaussianFilter()
    filt.prepare_from_grid("O16")
    return filt


class TestTransferFunction:
    @pytest.mark.parametrize("l, m", [(3, 2), (6, 3)])
    def test_matches_analytic_attenuation(self, o32, l, m):
        lat, lon = reduced_gaussian_grid("O32")
        y = real_harmonic(l, m, lat, lon)
        out = o32.compute(1, K, y)
        measured = np.sum(out * y) / np.sum(y * y)
        expected = analytic_attenuation(l)
        assert expected < 0.9                       # the test is not trivial
        assert abs(measured / expected - 1.0) < 2e-2, (measured, expected)
        shape_residual = np.sqrt(np.mean((out - measured * y) ** 2) / np.mean(y ** 2))
        assert shape_residual < 1e-2

    def test_constant_preserved(self, o32):
        out = o32.compute(1, K, np.full(o32._n2d, 3.0))
        np.testing.assert_allclose(out, 3.0, atol=1e-10)

    def test_noise_is_damped(self, o32):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(o32._n2d)
        out = o32.compute(1, 2 * math.pi / 1000.0, noise)
        assert np.all(np.isfinite(out))
        assert out.var() < 0.5 * noise.var()

    def test_mesh_sizes(self, o16):
        assert o16._n2d == 4 * 16 * (16 + 9) == 1600
        assert o16._e2d == 2 * 1600 - 4

    def test_data_is_nodal(self, o16):
        data = np.arange(o16._n2d, dtype=float)
        np.testing.assert_allclose(
            o16.compute(1, K, data, on="nodes"), o16.compute(1, K, data))
        # Nodal data is not element data...
        with pytest.raises(ValueError, match="element count"):
            o16.compute(1, K, data, on="elements")
        # ...and element-length data has no operator to be filtered with,
        # because meshtype='s' never assembles one.
        with pytest.raises(ValueError, match="filter_elements"):
            o16.compute(1, K, np.zeros(o16._e2d))


class TestEntryPointsAgree:
    def test_points_equal_grid(self, o16):
        lat, lon = reduced_gaussian_grid("O16")
        filt = ReducedGaussianFilter()
        filt.prepare_from_points(lat, lon)
        for attr in ("_ss", "_ii", "_jj", "_area"):
            np.testing.assert_array_equal(
                np.asarray(getattr(filt, attr)), np.asarray(getattr(o16, attr)))

    def test_points_accept_negative_longitudes(self, o16):
        lat, lon = reduced_gaussian_grid("O16")
        filt = ReducedGaussianFilter()
        filt.prepare_from_points(lat, np.where(lon >= 180.0, lon - 360.0, lon))
        assert filt._n2d == o16._n2d and filt._e2d == o16._e2d
        # Rounding of the shifted longitudes may re-triangulate the planar
        # polar-cap facets differently, so compare filtered fields loosely
        # rather than sparse triplets.
        lat_, lon_ = reduced_gaussian_grid("O16")
        y = real_harmonic(4, 2, lat_, lon_)
        np.testing.assert_allclose(
            filt.compute(1, K, y), o16.compute(1, K, y), atol=2e-2 * np.abs(y).max())

    @pytest.mark.parametrize("names", [
        ("latitude", "longitude"), ("lat", "lon"), ("latitudes", "longitudes")])
    def test_dataset_coordinates(self, o16, names):
        xr = pytest.importorskip("xarray")
        lat, lon = reduced_gaussian_grid("O16")
        ds = xr.Dataset(
            {"t2m": ("values", np.zeros(lat.size))},
            coords={names[0]: ("values", lat), names[1]: ("values", lon)})
        filt = ReducedGaussianFilter()
        filt.prepare_from_data_array(ds)
        np.testing.assert_array_equal(np.asarray(filt._ss), np.asarray(o16._ss))

    def test_dataarray(self, o16):
        xr = pytest.importorskip("xarray")
        lat, lon = reduced_gaussian_grid("O16")
        ds = xr.Dataset(
            {"t2m": ("values", np.zeros(lat.size))},
            coords={"latitude": ("values", lat), "longitude": ("values", lon)})
        filt = ReducedGaussianFilter()
        filt.prepare_from_data_array(ds["t2m"])
        np.testing.assert_array_equal(np.asarray(filt._ss), np.asarray(o16._ss))

    def test_coordinates_split_between_coords_and_data_vars(self, o16):
        xr = pytest.importorskip("xarray")
        lat, lon = reduced_gaussian_grid("O16")
        ds = xr.Dataset(
            {"longitude": ("values", lon)},
            coords={"latitude": ("values", lat)})
        filt = ReducedGaussianFilter()
        filt.prepare_from_data_array(ds)
        np.testing.assert_array_equal(np.asarray(filt._ss), np.asarray(o16._ss))

    def test_coordinates_as_data_variables(self, o16):
        xr = pytest.importorskip("xarray")
        lat, lon = reduced_gaussian_grid("O16")
        ds = xr.Dataset({"latitude": ("values", lat), "longitude": ("values", lon)})
        filt = ReducedGaussianFilter()
        filt.prepare_from_data_array(ds)
        np.testing.assert_array_equal(np.asarray(filt._ss), np.asarray(o16._ss))

    def test_two_dimensional_coordinates_point_to_latlonfilter(self):
        xr = pytest.importorskip("xarray")
        lon2, lat2 = np.meshgrid(np.arange(0.0, 360.0, 10.0), np.arange(-80.0, 81.0, 10.0))
        ds = xr.Dataset(coords={"latitude": (("y", "x"), lat2), "longitude": (("y", "x"), lon2)})
        with pytest.raises(ValueError, match="LatLonFilter"):
            ReducedGaussianFilter().prepare_from_data_array(ds)

    def test_unequal_1d_coordinates_point_to_latlonfilter(self):
        # A regular lat/lon grid stores its two axes separately, so 1-D
        # coordinates of different lengths are almost always a regular grid.
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(coords={
            "latitude": ("lat", np.linspace(-80.0, 80.0, 9)),
            "longitude": ("lon", np.arange(0.0, 360.0, 30.0)),
        })
        with pytest.raises(ValueError, match="LatLonFilter"):
            ReducedGaussianFilter().prepare_from_data_array(ds)

    def test_scalar_coordinates_do_not_blame_latlonfilter(self):
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(
            {"t2m": ("values", np.zeros(10))},
            coords={"latitude": 45.0, "longitude": 10.0})
        with pytest.raises(ValueError) as excinfo:
            ReducedGaussianFilter().prepare_from_data_array(ds)
        message = str(excinfo.value)
        assert "LatLonFilter" not in message
        assert "0-D" in message

    def test_missing_coordinates(self):
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset({"t2m": ("values", np.zeros(10))})
        with pytest.raises(ValueError, match="latitude"):
            ReducedGaussianFilter().prepare_from_data_array(ds)

    def test_netcdf_file(self, tmp_path, o16):
        xr = pytest.importorskip("xarray")
        lat, lon = reduced_gaussian_grid("O16")
        ds = xr.Dataset(
            {"t2m": ("values", np.zeros(lat.size))},
            coords={"latitude": ("values", lat), "longitude": ("values", lon)})
        path = tmp_path / "o16.nc"
        ds.to_netcdf(path)
        filt = ReducedGaussianFilter()
        filt.prepare_from_file(str(path))
        np.testing.assert_array_equal(np.asarray(filt._ss), np.asarray(o16._ss))

    def test_grib_file(self, tmp_path):
        """A GRIB message read through cfgrib yields the same grid as the name."""
        eccodes = pytest.importorskip("eccodes")
        pytest.importorskip("cfgrib")
        pytest.importorskip("xarray")
        h = eccodes.codes_grib_new_from_samples("reduced_gg_pl_32_grib2")
        path = tmp_path / "n32.grib"
        try:
            with open(path, "wb") as fh:
                eccodes.codes_write(h, fh)
        finally:
            eccodes.codes_release(h)
        from_file = ReducedGaussianFilter()
        from_file.prepare_from_file(str(path), engine="cfgrib")
        from_name = ReducedGaussianFilter()
        from_name.prepare_from_grid("N32")
        assert from_file._n2d == from_name._n2d == 6114
        # cfgrib's coordinates differ from ours by ~1e-12 degrees, which may
        # flip the diagonal of coplanar quads between equal-length rows, so
        # compare filtered fields rather than sparse triplets.
        lat, lon = reduced_gaussian_grid("N32")
        y = real_harmonic(3, 2, lat, lon)
        np.testing.assert_allclose(
            from_file.compute(1, K, y), from_name.compute(1, K, y),
            atol=2e-2 * np.abs(y).max())


class TestMask:
    def test_valid_constant_preserved_and_masked_points_untouched(self):
        lat, lon = reduced_gaussian_grid("O16")
        valid = lat > 0.0                                   # southern hemisphere is "land"
        filt = ReducedGaussianFilter()
        filt.prepare_from_points(lat, lon, mask=valid)
        # A conspicuous fill value: masked points must come back untouched,
        # not merely zeroed, and must not leak into the valid constant.
        data = np.where(valid, 4.0, -7.5)
        out = filt.compute(1, 2 * math.pi / 3000.0, data)
        np.testing.assert_allclose(out[valid], 4.0, atol=1e-8)
        np.testing.assert_array_equal(out[~valid], -7.5)

    def test_random_field_does_not_leak_into_masked_points(self):
        lat, lon = reduced_gaussian_grid("O16")
        valid = np.abs(lat) < 60.0
        filt = ReducedGaussianFilter()
        filt.prepare_from_grid("O16", mask=valid)
        rng = np.random.default_rng(3)
        data = np.where(valid, rng.standard_normal(lat.size), -7.5)
        out = filt.compute(1, 2 * math.pi / 2000.0, data)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out[~valid], -7.5)
        assert out[valid].var() < data[valid].var()

    def test_mask_shape_is_checked(self):
        lat, lon = reduced_gaussian_grid("O16")
        with pytest.raises(ValueError, match="mask"):
            ReducedGaussianFilter().prepare_from_points(lat, lon, mask=np.ones(7, dtype=bool))

    def test_float_mask_with_nan_is_rejected(self):
        # bool(nan) is True, so ~np.isnan() forgotten on a float field would
        # silently mark the invalid points valid.
        lat, lon = reduced_gaussian_grid("O16")
        mask = np.ones(lat.size)
        mask[0] = np.nan
        with pytest.raises(ValueError, match="boolean mask"):
            ReducedGaussianFilter().prepare_from_points(lat, lon, mask=mask)

    def test_finite_float_mask_matches_the_bool_mask(self):
        lat, lon = reduced_gaussian_grid("O16")
        valid = lat > 0.0
        as_bool = ReducedGaussianFilter()
        as_bool.prepare_from_points(lat, lon, mask=valid)
        as_float = ReducedGaussianFilter()
        as_float.prepare_from_points(lat, lon, mask=valid.astype(float))
        np.testing.assert_array_equal(np.asarray(as_float._ss), np.asarray(as_bool._ss))


class TestInheritedCapabilities:
    def test_save_load_round_trip(self, tmp_path, o16):
        o16.save_to_file(str(tmp_path / "cache"))
        loaded = ReducedGaussianFilter.load_from_file(str(tmp_path / "cache.npz"))
        rng = np.random.default_rng(1)
        data = rng.standard_normal(o16._n2d)
        np.testing.assert_allclose(
            loaded.compute(1, K, data), o16.compute(1, K, data), rtol=1e-12, atol=1e-12)

    def test_velocity(self, o16):
        rng = np.random.default_rng(4)
        u, v = o16.compute_velocity(
            1, K, rng.standard_normal(o16._n2d), rng.standard_normal(o16._n2d))
        assert u.shape == v.shape == (o16._n2d,)
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))

    def test_spectra(self, o16):
        lat, lon = reduced_gaussian_grid("O16")
        data = real_harmonic(4, 1, lat, lon) + 0.1 * real_harmonic(10, 5, lat, lon)
        ks = 2 * math.pi / np.array([1000.0, 3000.0, 9000.0])
        spectra = o16.compute_spectra_scalar(1, ks, data)
        assert spectra.shape == (4,)
        assert np.all(np.isfinite(spectra))
        area = np.asarray(o16._area)
        np.testing.assert_allclose(spectra[0], np.sum(area * data ** 2) / area.sum())
        assert np.all(np.diff(spectra[1:]) >= 0)            # larger scales remove more

    def test_prepare_signature_has_no_deprecated_gpu_argument(self):
        import inspect
        for name in ("prepare_from_grid", "prepare_from_points",
                     "prepare_from_data_array", "prepare_from_file"):
            params = inspect.signature(getattr(ReducedGaussianFilter, name)).parameters
            assert "gpu" not in params
            assert "mask" in params and "full" in params
            if name == "prepare_from_grid":
                assert "check_coverage" not in params
            else:
                assert "check_coverage" in params


class TestErrors:
    def test_unknown_grid(self):
        with pytest.raises(ValueError, match="grid name"):
            ReducedGaussianFilter().prepare_from_grid("Q96")

    def test_unequal_1d_points_point_to_latlonfilter(self):
        # The axes of a regular lat/lon grid, not a point list.
        with pytest.raises(ValueError, match="LatLonFilter"):
            ReducedGaussianFilter().prepare_from_points(
                np.linspace(-80.0, 80.0, 9), np.arange(0.0, 360.0, 30.0))

    def test_regional_points(self):
        lon, lat = np.meshgrid(np.linspace(0.0, 10.0, 6), np.linspace(0.0, 10.0, 6))
        with pytest.raises(ValueError, match="sphere"):
            ReducedGaussianFilter().prepare_from_points(lat.ravel(), lon.ravel())

    def test_check_coverage_passthrough(self):
        """A regular lat-lon point set is rejected by the heuristic, and passes
        without it."""
        lats = np.arange(88.75, -88.76, -2.5)
        lons = np.arange(0.0, 360.0, 2.5)
        lat = np.repeat(lats, lons.size)
        lon = np.tile(lons, lats.size)
        with pytest.raises(ValueError, match="check_coverage"):
            ReducedGaussianFilter().prepare_from_points(lat, lon)
        filt = ReducedGaussianFilter()
        filt.prepare_from_points(lat, lon, check_coverage=False)
        assert filt._n2d == lat.size
        out = filt.compute(1, 2 * math.pi / 3000.0, np.full(lat.size, 2.0))
        np.testing.assert_allclose(out, 2.0, atol=1e-8)

    def test_exported(self):
        assert implicit_filter.ReducedGaussianFilter is ReducedGaussianFilter
