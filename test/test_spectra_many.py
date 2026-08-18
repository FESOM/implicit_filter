"""The batch spectra methods must agree with the per-snapshot loop."""
import numpy as np
import pytest

from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


def build_filter(Lx=20):
    xx = np.arange(0, Lx + 1, dtype=float)
    nx = len(xx)
    nodnum = np.reshape(np.arange(nx * nx), [nx, nx]).T
    xcoord = np.zeros((nx, nx))
    ycoord = xcoord.copy()
    for i in range(nx):
        ycoord[i, :] = xx
        xcoord[:, i] = xx
    tri = make_tri(nodnum, nx, nx)
    n2d, e2d = nx * nx, len(tri)
    filt = TriangularFilter()
    filt.prepare(n2d, e2d, tri, xcoord.flatten(), ycoord.flatten(),
                 meshtype='m', cartesian=True, full=False)
    return filt, n2d, e2d


class TestScalarMany:

    @pytest.mark.parametrize("highpass", [True, False])
    @pytest.mark.parametrize("demean", [False, True])
    def test_matches_per_snapshot_loop(self, highpass, demean):
        filt, n2d, _ = build_filter()
        rng = np.random.default_rng(0)
        data = rng.normal(size=(3, n2d)) + 4.0
        k = [3.0, 8.0]

        many = filt.compute_spectra_scalar_many(
            2, k, data, highpass=highpass, demean=demean)
        for t in range(data.shape[0]):
            one = filt.compute_spectra_scalar(
                2, k, data[t], highpass=highpass, demean=demean)
            np.testing.assert_allclose(many[t], one, rtol=1e-8, atol=1e-12)

    def test_does_not_mutate_input(self):
        filt, n2d, _ = build_filter()
        data = np.random.default_rng(1).normal(size=(2, n2d)) + 7.0
        orig = data.copy()
        filt.compute_spectra_scalar_many(2, [5.0], data, demean=True)
        np.testing.assert_array_equal(data, orig)

    def test_mask_is_applied(self):
        filt, n2d, _ = build_filter()
        data = np.random.default_rng(2).normal(size=(2, n2d))
        mask = np.zeros(n2d, dtype=bool)
        mask[: n2d // 4] = True
        a = filt.compute_spectra_scalar_many(2, [5.0], data)
        b = filt.compute_spectra_scalar_many(2, [5.0], data, mask=mask)
        assert not np.allclose(a, b)

    def test_rejects_1d_input(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError, match="2-D"):
            filt.compute_spectra_scalar_many(2, [5.0],
                                             np.zeros(n2d))

    def test_shape(self):
        filt, n2d, _ = build_filter()
        data = np.zeros((4, n2d))
        out = filt.compute_spectra_scalar_many(2, [3.0, 5.0, 9.0], data)
        assert out.shape == (4, 4)


class TestVelocityMany:

    @pytest.mark.parametrize("highpass", [True, False])
    def test_matches_per_snapshot_loop(self, highpass):
        filt, n2d, _ = build_filter()
        rng = np.random.default_rng(3)
        ux = rng.normal(size=(3, n2d))
        vy = rng.normal(size=(3, n2d))
        k = [3.0, 8.0]

        many = filt.compute_spectra_velocity_many(
            2, k, ux, vy, highpass=highpass)
        for t in range(ux.shape[0]):
            one = filt.compute_spectra_velocity(
                2, k, ux[t], vy[t], highpass=highpass)
            np.testing.assert_allclose(many[t], one, rtol=1e-8, atol=1e-12)

    def test_rejects_mismatched_shapes(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError):
            filt.compute_spectra_velocity_many(
                2, [5.0], np.zeros((2, n2d)), np.zeros((3, n2d)))


class TestPreconditionerSwitching:

    def test_preconditioner_restored(self):
        filt, n2d, _ = build_filter()
        data = np.random.default_rng(4).normal(size=(2, n2d))
        filt.set_preconditioner("jacobi")
        filt.compute_spectra_scalar_many(
            2, [0.5, 5.0], data, vcycle_above=1.0)
        assert filt.get_preconditioner() == "jacobi"

    def test_switching_does_not_change_the_result(self):
        filt, n2d, _ = build_filter()
        data = np.random.default_rng(5).normal(size=(2, n2d))
        k = [0.5, 5.0]
        plain = filt.compute_spectra_scalar_many(2, k, data)
        switched = filt.compute_spectra_scalar_many(
            2, k, data, vcycle_above=1.0)
        np.testing.assert_allclose(plain, switched, rtol=1e-5)