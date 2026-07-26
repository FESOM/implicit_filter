"""
Tests for LatLonFilter, in particular the masked (land) configuration.
"""
import numpy as np
import pytest
from implicit_filter import LatLonFilter


def build_grid(nlon=9, nlat=8):
    lon = np.linspace(0.0, 20.0, nlon)
    lat = np.linspace(-10.0, 10.0, nlat)
    return lon, lat


class TestMaskedPrepare:
    """A land mask must produce a usable filter, not a broken one."""

    def test_masked_sparse_arrays_stay_aligned(self):
        lon, lat = build_grid()
        mask = np.ones((len(lon), len(lat)), dtype=bool)
        mask[4, 4] = False  # one land point

        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True, mask=mask)

        assert len(filt._ss) == len(filt._ii) == len(filt._jj), (
            "sparse triplet arrays must have equal length after masking; "
            f"got ss={len(filt._ss)} ii={len(filt._ii)} jj={len(filt._jj)}"
        )

    def test_masked_compute_runs(self):
        lon, lat = build_grid()
        mask = np.ones((len(lon), len(lat)), dtype=bool)
        mask[4, 4] = False

        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True, mask=mask)

        np.random.seed(0)
        data = np.random.randn(len(lon), len(lat))
        result = filt.compute(1, 0.5, data)

        assert result.shape == (len(lon), len(lat))
        assert np.all(np.isfinite(result))

    def test_masked_constant_preserved(self):
        lon, lat = build_grid()
        mask = np.ones((len(lon), len(lat)), dtype=bool)
        mask[4, 4] = False

        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True, mask=mask)

        result = filt.compute(1, 0.5, np.full((len(lon), len(lat)), 7.0))
        np.testing.assert_allclose(result, 7.0, atol=1e-5)

    def test_masked_velocity_runs(self):
        lon, lat = build_grid()
        mask = np.ones((len(lon), len(lat)), dtype=bool)
        mask[2, 3] = False

        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True, mask=mask)

        np.random.seed(1)
        ux = np.random.randn(len(lon), len(lat))
        vy = np.random.randn(len(lon), len(lat))
        ux_f, vy_f = filt.compute_velocity(1, 0.5, ux, vy)

        assert ux_f.shape == ux.shape and vy_f.shape == vy.shape
        assert np.all(np.isfinite(ux_f)) and np.all(np.isfinite(vy_f))


class TestUnmaskedStillWorks:
    """The unmasked path must not regress."""

    def test_unmasked_constant_preserved(self):
        lon, lat = build_grid()
        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True)

        result = filt.compute(1, 0.5, np.full((len(lon), len(lat)), 3.0))
        np.testing.assert_allclose(result, 3.0, atol=1e-5)

    def test_unmasked_variance_decreases(self):
        lon, lat = build_grid()
        filt = LatLonFilter()
        filt.prepare(lat, lon, cartesian=True, local=True)

        np.random.seed(2)
        data = np.random.randn(len(lon), len(lat))
        result = filt.compute(1, 0.5, data)
        assert np.var(result) <= np.var(data) + 1e-10
