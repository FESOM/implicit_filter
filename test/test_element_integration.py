"""
Integration tests for element-based filtering combined with other features:
variable-scale filtering, mesh masking, spectra computation, and velocity fields.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


# ---------------------------------------------------------------------------
# Shared mesh builder
# ---------------------------------------------------------------------------

def build_filter(Lx=20, mask=None, filter_elements=True):
    """Build a prepared TriangularFilter on a regular triangular mesh."""
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
    n2d = len(xcoord)
    e2d = len(tri)

    if mask is None:
        mask = np.ones(e2d, dtype=bool)

    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=False,
        mask=mask, filter_elements=filter_elements,
    )
    return filt, n2d, e2d, tri, xcoord, ycoord


# ===========================================================================
# 1. Variable-scale element filtering
# ===========================================================================

class TestVariableScaleElements:
    """Element filtering with spatially-varying filter scale (array k)."""

    def test_uniform_array_matches_scalar(self):
        """Array of identical scales should give the same result as a scalar."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)

        result_scalar = filt.compute(1, 5.0, data)
        result_array = filt.compute(1, np.full(e2d, 5.0), data)

        np.testing.assert_allclose(result_array, result_scalar, atol=1e-6,
            err_msg="Uniform array k should match scalar k for elements")

    def test_variable_scale_runs(self):
        """Non-uniform scale array should run without error."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        # Linearly varying scale
        k = np.linspace(2.0, 10.0, e2d)

        result = filt.compute(1, k, data)
        assert result.shape == (e2d,)
        assert np.all(np.isfinite(result))

    def test_variable_scale_reduces_variance(self):
        """Variable-scale filtering should still reduce variance overall."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k = np.linspace(2.0, 10.0, e2d)

        result = filt.compute(1, k, data)
        assert np.var(result) <= np.var(data) + 1e-10

    def test_small_scale_region_smoother(self):
        """
        Elements with small k (aggressive smoothing) should be smoother
        than elements with large k (mild smoothing).
        """
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)

        # Half gets small scale, half gets large
        k = np.ones(e2d) * 20.0  # Large (mild smoothing)
        k[:e2d // 2] = 2.0       # Small (aggressive smoothing, first half)

        result = filt.compute(1, k, data)

        var_small_k = np.var(result[:e2d // 2])
        var_large_k = np.var(result[e2d // 2:])

        # The half with small k should be smoother
        assert var_small_k < var_large_k, \
            f"Small-k region should be smoother: var={var_small_k:.4f} vs {var_large_k:.4f}"

    def test_variable_scale_velocity_elements(self):
        """Variable-scale element velocity filtering should work."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        ux = np.random.randn(e2d)
        vy = np.random.randn(e2d)
        k = np.linspace(2.0, 10.0, e2d)

        ux_f, vy_f = filt.compute_velocity(1, k, ux, vy)
        assert ux_f.shape == (e2d,)
        assert vy_f.shape == (e2d,)
        assert np.all(np.isfinite(ux_f))
        assert np.all(np.isfinite(vy_f))

        # Should reduce kinetic energy
        ke_before = np.mean(ux**2 + vy**2)
        ke_after = np.mean(ux_f**2 + vy_f**2)
        assert ke_after <= ke_before + 1e-10


# ===========================================================================
# 2. Masked mesh + element filtering
# ===========================================================================

class TestMaskedElementFiltering:
    """Element filtering on a mesh with masked (land) regions."""

    @staticmethod
    def _make_half_masked_filter(Lx=20):
        """Create a mesh where the first half of elements are masked (land)."""
        xx = np.arange(0, Lx + 1, dtype=float)
        nx = len(xx)
        nodnum = np.reshape(np.arange(nx * nx), [nx, nx]).T
        xcoord = np.zeros((nx, nx))
        ycoord = xcoord.copy()
        for i in range(nx):
            ycoord[i, :] = xx
            xcoord[:, i] = xx
        tri = make_tri(nodnum, nx, nx)
        n2d = nx * nx
        e2d = len(tri)

        mask = np.ones(e2d, dtype=bool)
        mask[:e2d // 2] = False  # first half is land

        return build_filter(Lx, mask=mask)

    def test_masked_element_scalar(self):
        """Element scalar filtering works with partial mask."""
        filt, _, e2d, *_ = self._make_half_masked_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        result = filt.compute(1, 5.0, data)
        assert result.shape == (e2d,)
        assert np.all(np.isfinite(result))

    def test_masked_element_velocity(self):
        """Element velocity filtering works with partial mask."""
        filt, _, e2d, *_ = self._make_half_masked_filter()
        np.random.seed(42)
        ux = np.random.randn(e2d)
        vy = np.random.randn(e2d)
        ux_f, vy_f = filt.compute_velocity(1, 5.0, ux, vy)
        assert ux_f.shape == (e2d,)
        assert vy_f.shape == (e2d,)
        assert np.all(np.isfinite(ux_f))

    def test_masked_constant_preserved(self):
        """Constant field should be preserved even on a masked mesh."""
        filt, _, e2d, *_ = self._make_half_masked_filter()
        data = np.full(e2d, 7.0)
        result = filt.compute(1, 5.0, data)
        np.testing.assert_allclose(result, 7.0, atol=1e-5,
            err_msg="Constant field not preserved on masked mesh")

    def test_masked_variable_scale(self):
        """Variable scale + mask combination should work."""
        filt, _, e2d, *_ = self._make_half_masked_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k = np.linspace(2.0, 10.0, e2d)

        result = filt.compute(1, k, data)
        assert result.shape == (e2d,)
        assert np.all(np.isfinite(result))

    def test_masked_zero_field(self):
        """Zero field should remain zero even with masking."""
        filt, _, e2d, *_ = self._make_half_masked_filter()
        result = filt.compute(1, 5.0, np.zeros(e2d))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ===========================================================================
# 3. Element-based spectra computation
# ===========================================================================

class TestElementSpectra:
    """Spectra computation using element-based data."""

    def test_scalar_spectra_shape(self):
        """Spectra output has correct shape: len(k) + 1."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k_list = [2.0, 5.0, 10.0]

        spectra = filt.compute_spectra_scalar(1, k_list, data)
        assert spectra.shape == (len(k_list) + 1,)

    def test_scalar_spectra_total_variance(self):
        """First entry of spectra should be (area-weighted) total variance."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k_list = [5.0]

        spectra = filt.compute_spectra_scalar(1, k_list, data)
        # spectra[0] is area-weighted mean of data², should be positive
        assert spectra[0] > 0

    def test_scalar_spectra_positive(self):
        """All spectral entries should be non-negative."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k_list = [2.0, 5.0, 10.0]

        spectra = filt.compute_spectra_scalar(1, k_list, data)
        assert np.all(spectra >= -1e-10), \
            f"Negative spectral entry: {spectra}"

    def test_velocity_spectra_shape(self):
        """Velocity spectra output has correct shape."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        ux = np.random.randn(e2d)
        vy = np.random.randn(e2d)
        k_list = [2.0, 5.0]

        spectra = filt.compute_spectra_velocity(1, k_list, ux, vy)
        assert spectra.shape == (len(k_list) + 1,)

    def test_velocity_spectra_positive(self):
        """KE spectral entries should be non-negative."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        ux = np.random.randn(e2d)
        vy = np.random.randn(e2d)
        k_list = [2.0, 5.0, 10.0]

        spectra = filt.compute_spectra_velocity(1, k_list, ux, vy)
        assert np.all(spectra >= -1e-10), \
            f"Negative velocity spectral entry: {spectra}"

    def test_spectra_with_mask(self):
        """Spectra with a mask applied during computation."""
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)
        k_list = [5.0]

        # Mask out first quarter of elements for spectra computation
        mask = np.zeros(e2d, dtype=bool)
        mask[:e2d // 4] = True  # Exclude these from spectra

        spectra = filt.compute_spectra_scalar(1, k_list, data, mask=mask)
        assert spectra.shape == (2,)
        assert spectra[0] > 0


# ===========================================================================
# 4. Simultaneous node and element filtering on the same filter
# ===========================================================================

class TestNodeElementCoexistence:
    """A single filter instance can handle both node and element data."""

    def test_node_then_element(self):
        """Filtering nodes then elements on the same prepared filter."""
        filt, n2d, e2d, *_ = build_filter()
        np.random.seed(42)

        node_data = np.random.randn(n2d)
        elem_data = np.random.randn(e2d)

        r_node = filt.compute(1, 5.0, node_data)
        r_elem = filt.compute(1, 5.0, elem_data)

        assert r_node.shape == (n2d,)
        assert r_elem.shape == (e2d,)
        assert np.var(r_node) < np.var(node_data)
        assert np.var(r_elem) < np.var(elem_data)

    def test_element_then_node(self):
        """Reverse order: elements first, then nodes."""
        filt, n2d, e2d, *_ = build_filter()
        np.random.seed(42)

        elem_data = np.random.randn(e2d)
        node_data = np.random.randn(n2d)

        r_elem = filt.compute(1, 5.0, elem_data)
        r_node = filt.compute(1, 5.0, node_data)

        assert r_elem.shape == (e2d,)
        assert r_node.shape == (n2d,)

    def test_node_velocity_and_element_scalar(self):
        """Mix velocity filtering on nodes with scalar on elements."""
        filt, n2d, e2d, *_ = build_filter()
        np.random.seed(42)

        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        elem_data = np.random.randn(e2d)

        ux_f, vy_f = filt.compute_velocity(1, 5.0, ux, vy)
        r_elem = filt.compute(1, 5.0, elem_data)

        assert ux_f.shape == (n2d,)
        assert r_elem.shape == (e2d,)


# ===========================================================================
# 5. Element filtering with different filter orders
# ===========================================================================

class TestElementFilterOrders:
    """Element filtering with various filter orders."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_produces_valid_output(self, order):
        filt, _, e2d, *_ = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)

        result = filt.compute(order, 5.0, data)
        assert result.shape == (e2d,)
        assert np.all(np.isfinite(result))
        assert np.var(result) < np.var(data)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_constant_preserved(self, order):
        filt, _, e2d, *_ = build_filter()
        data = np.full(e2d, 3.14)
        result = filt.compute(order, 5.0, data)
        np.testing.assert_allclose(result, 3.14, atol=1e-5)
