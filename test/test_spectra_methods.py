"""
Tests for the spectra methods: gamma/highpass plumbing, demean behaviour,
cross-spectrum self-consistency, and higher-order smoke tests.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


def build_filter(Lx=20, filter_elements=True):
    xx = np.arange(0, Lx + 1, dtype=float)
    yy = np.arange(0, Lx + 1, dtype=float)
    nx, ny = len(xx), len(yy)
    nodnum = np.reshape(np.arange(nx * ny), [ny, nx]).T
    xcoord = np.zeros((nx, ny))
    ycoord = xcoord.copy()
    for i in range(nx):
        ycoord[i, :] = yy
    for i in range(ny):
        xcoord[:, i] = xx
    xcoord = xcoord.flatten()
    ycoord = ycoord.flatten()
    tri = make_tri(nodnum, nx, ny)
    n2d, e2d = len(xcoord), len(tri)
    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=False,
        filter_elements=filter_elements,
    )
    return filt, n2d, e2d


# ---------------------------------------------------------------------------
# 1. gamma / highpass actually reach the filter inside the spectra loop
# ---------------------------------------------------------------------------

class TestSpectraGammaPlumbing:
    """gamma/highpass must change the spectra output (regression: they were ignored)."""

    def test_velocity_gamma_changes_result(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        k = [3.0, 8.0]

        s_g2 = filt.compute_spectra_velocity(2, k, ux, vy, gamma=2.0)
        s_g05 = filt.compute_spectra_velocity(2, k, ux, vy, gamma=0.5)
        assert not np.allclose(s_g2, s_g05), \
            "gamma is not reaching the filter inside compute_spectra_velocity"

    def test_scalar_gamma_changes_result(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)
        k = [3.0, 8.0]

        s_g2 = filt.compute_spectra_scalar(2, k, data, gamma=2.0)
        s_g05 = filt.compute_spectra_scalar(2, k, data, gamma=0.5)
        assert not np.allclose(s_g2, s_g05), \
            "gamma is not reaching the filter inside compute_spectra_scalar"

    def test_highpass_lowpass_differ(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)
        k = [5.0]

        hp = filt.compute_spectra_scalar(2, k, data, highpass=True)
        lp = filt.compute_spectra_scalar(2, k, data, highpass=False)
        assert not np.allclose(hp, lp), \
            "highpass and lowpass spectra should differ"


# ---------------------------------------------------------------------------
# 2. Cross-spectrum self-consistency: cross(field, field) == auto-spectrum
# ---------------------------------------------------------------------------

class TestCrossSelfConsistency:
    """Cross-spectrum of a field with itself equals its ordinary power spectrum."""

    def test_cross_velocity_equals_velocity(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        k = [3.0, 8.0, 15.0]

        auto = filt.compute_spectra_velocity(2, k, ux, vy)
        cross = filt.compute_spectra_cross_velocity(2, k, ux, vy, ux, vy)
        np.testing.assert_allclose(cross, auto, rtol=1e-5, atol=1e-7,
            err_msg="cross_velocity(u,v,u,v) != velocity auto-spectrum")

    def test_cross_scalar_equals_scalar(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)
        k = [3.0, 8.0, 15.0]

        auto = filt.compute_spectra_scalar(2, k, data)
        cross = filt.compute_spectra_cross_scalar(2, k, data, data)
        np.testing.assert_allclose(cross, auto, rtol=1e-5, atol=1e-7,
            err_msg="cross_scalar(data,data) != scalar auto-spectrum")


# ---------------------------------------------------------------------------
# 3. Demean: changes output when a mean is present, and does not mutate input
# ---------------------------------------------------------------------------

class TestDemean:
    """demean must remove a constant offset and must not mutate the caller's arrays."""

    def test_demean_changes_offset_field(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d) + 10.0   # large constant offset
        k = [5.0]

        s_no = filt.compute_spectra_scalar(2, k, data, demean=False)
        s_yes = filt.compute_spectra_scalar(2, k, data, demean=True)
        # The total-variance bin [0] should drop a lot once the mean is removed.
        assert s_yes[0] < s_no[0], \
            "demean should reduce the total-variance bin for an offset field"

    def test_demean_does_not_mutate_input(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        ux = np.random.randn(n2d) + 5.0
        vy = np.random.randn(n2d) - 3.0
        ux_orig = ux.copy()
        vy_orig = vy.copy()

        filt.compute_spectra_velocity(2, [5.0], ux, vy, demean=True)
        np.testing.assert_array_equal(ux, ux_orig,
            err_msg="compute_spectra_velocity mutated ux when demean=True")
        np.testing.assert_array_equal(vy, vy_orig,
            err_msg="compute_spectra_velocity mutated vy when demean=True")

    def test_demean_scalar_does_not_mutate_input(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d) + 7.0
        data_orig = data.copy()

        filt.compute_spectra_scalar(2, [5.0], data, demean=True)
        np.testing.assert_array_equal(data, data_orig,
            err_msg="compute_spectra_scalar mutated data when demean=True")


# ---------------------------------------------------------------------------
# 4. Higher-order spectra smoke tests (n=3,4 run and give finite, sane output)
# ---------------------------------------------------------------------------

class TestHigherOrderSpectra:
    """n=3,4 spectra must run, be finite, and be non-negative in the auto case."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_velocity_spectra_finite_nonneg(self, n):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        k = [3.0, 8.0]

        s = filt.compute_spectra_velocity(n, k, ux, vy)
        assert s.shape == (len(k) + 1,)
        assert np.all(np.isfinite(s)), f"non-finite spectra for n={n}"
        assert np.all(s >= -1e-7), f"negative spectral entry for n={n}: {s}"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_scalar_spectra_finite_nonneg(self, n):
        filt, _, e2d = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(e2d)   # element-based field
        k = [3.0, 8.0]

        s = filt.compute_spectra_scalar(n, k, data)
        assert s.shape == (len(k) + 1,)
        assert np.all(np.isfinite(s)), f"non-finite spectra for n={n}"
        assert np.all(s >= -1e-7), f"negative spectral entry for n={n}: {s}"


# ---------------------------------------------------------------------------
# 5. Mask handling: masked points do not enter the spectrum
# ---------------------------------------------------------------------------

class TestMaskInSpectra:
    """A mask should change the spectrum and exclude the masked region."""

    def test_mask_changes_spectrum(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)
        k = [5.0]

        full = filt.compute_spectra_scalar(2, k, data)
        mask = np.zeros(n2d, dtype=bool)
        mask[: n2d // 2] = True   # exclude first half
        masked = filt.compute_spectra_scalar(2, k, data, mask=mask)
        assert not np.allclose(full, masked), \
            "masking did not change the spectrum"