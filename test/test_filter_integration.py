"""
Integration tests for TriangularFilter: full prepare() → compute() pipeline.
Verifies mathematical properties that any valid spatial filter must satisfy.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def build_filter(Lx=20, dxm=1, filter_elements=True, full=False):
    """Build a prepared TriangularFilter on a regular triangular mesh."""
    xx = np.arange(0, Lx + 1, dxm, dtype=float)
    yy = np.arange(0, Lx + 1, dxm, dtype=float)
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

    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=full,
        filter_elements=filter_elements,
    )
    return filt, n2d, e2d


@pytest.fixture
def filt_20():
    """20×20 mesh with element filtering enabled."""
    f, n2d, e2d = build_filter(20)
    return f, n2d, e2d


# ---------------------------------------------------------------------------
# 1. Constant field preservation
# ---------------------------------------------------------------------------

class TestConstantFieldPreservation:
    """Filtering a constant field must return that exact constant."""

    def test_constant_scalar_on_nodes(self, filt_20):
        filt, n2d, _ = filt_20
        const = 42.0
        data = np.full(n2d, const)
        result = filt.compute(1, 5.0, data)
        np.testing.assert_allclose(result, const, atol=1e-6,
            err_msg="Constant nodal scalar not preserved")

    def test_constant_scalar_on_elements(self, filt_20):
        filt, _, e2d = filt_20
        const = -7.5
        data = np.full(e2d, const)
        result = filt.compute(1, 5.0, data)
        np.testing.assert_allclose(result, const, atol=1e-6,
            err_msg="Constant element scalar not preserved")

    def test_constant_velocity(self, filt_20):
        filt, n2d, _ = filt_20
        ux = np.full(n2d, 3.0)
        vy = np.full(n2d, -2.0)
        ux_f, vy_f = filt.compute_velocity(1, 5.0, ux, vy)
        np.testing.assert_allclose(ux_f, 3.0, atol=1e-6)
        np.testing.assert_allclose(vy_f, -2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Zero field
# ---------------------------------------------------------------------------

class TestZeroField:
    """Filtering zeros must return zeros."""

    def test_zero_scalar_nodes(self, filt_20):
        filt, n2d, _ = filt_20
        result = filt.compute(1, 5.0, np.zeros(n2d))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_zero_scalar_elements(self, filt_20):
        filt, _, e2d = filt_20
        result = filt.compute(1, 5.0, np.zeros(e2d))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_zero_velocity(self, filt_20):
        filt, n2d, _ = filt_20
        ux_f, vy_f = filt.compute_velocity(1, 5.0, np.zeros(n2d), np.zeros(n2d))
        np.testing.assert_allclose(ux_f, 0.0, atol=1e-10)
        np.testing.assert_allclose(vy_f, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Variance reduction (smoothing never amplifies)
# ---------------------------------------------------------------------------

class TestVarianceReduction:
    """Filtering must never increase total variance."""

    def test_scalar_variance_decreases_nodes(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        data = np.random.randn(n2d)
        result = filt.compute(1, 5.0, data)
        assert np.var(result) <= np.var(data) + 1e-10, \
            f"Variance increased: {np.var(result):.6f} > {np.var(data):.6f}"

    def test_scalar_variance_decreases_elements(self, filt_20):
        filt, _, e2d = filt_20
        np.random.seed(42)
        data = np.random.randn(e2d)
        result = filt.compute(1, 5.0, data)
        assert np.var(result) <= np.var(data) + 1e-10, \
            f"Variance increased: {np.var(result):.6f} > {np.var(data):.6f}"

    def test_velocity_energy_decreases(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        ux_f, vy_f = filt.compute_velocity(1, 5.0, ux, vy)
        ke_before = np.mean(ux**2 + vy**2)
        ke_after = np.mean(ux_f**2 + vy_f**2)
        assert ke_after <= ke_before + 1e-10, \
            f"KE increased: {ke_after:.6f} > {ke_before:.6f}"


# ---------------------------------------------------------------------------
# 4. Monotonicity: larger scale → more smoothing → less variance
# ---------------------------------------------------------------------------

class TestScaleMonotonicity:
    """Increasing filter scale must decrease (or maintain) variance."""

    def test_increasing_scale_reduces_variance(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        data = np.random.randn(n2d)

        scales = [1.0, 3.0, 8.0, 20.0]
        variances = []
        for k in scales:
            result = filt.compute(1, k, data)
            variances.append(np.var(result))

        # Overall trend: smallest scale should have least variance,
        # largest scale should have most (closest to original)
        assert variances[-1] >= variances[0] - 1e-10, \
            f"Large scale should have more variance than small: " \
            f"k={scales[-1]}→var={variances[-1]:.6f}, k={scales[0]}→var={variances[0]:.6f}"


# ---------------------------------------------------------------------------
# 5. Filter order effect
# ---------------------------------------------------------------------------

class TestFilterOrder:
    """Both filter orders produce valid smoothing (reduce variance vs input)."""

    def test_both_orders_reduce_variance(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        data = np.random.randn(n2d)
        original_var = np.var(data)

        r1 = filt.compute(1, 3.0, data)
        r2 = filt.compute(2, 3.0, data)

        assert np.var(r1) < original_var, \
            f"Order-1 didn't reduce variance: {np.var(r1):.6f} >= {original_var:.6f}"
        assert np.var(r2) < original_var, \
            f"Order-2 didn't reduce variance: {np.var(r2):.6f} >= {original_var:.6f}"


# ---------------------------------------------------------------------------
# 6. Scalar vs velocity consistency
# ---------------------------------------------------------------------------

class TestScalarVelocityConsistency:
    """compute_velocity(n, k, ux, zeros) ≈ (compute(n, k, ux), zeros)."""

    def test_velocity_x_matches_scalar(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        data = np.random.randn(n2d)
        zeros = np.zeros(n2d)

        scalar_result = filt.compute(1, 5.0, data)
        ux_f, vy_f = filt.compute_velocity(1, 5.0, data, zeros)

        np.testing.assert_allclose(ux_f, scalar_result, atol=1e-6,
            err_msg="Velocity x-component differs from scalar filter")
        np.testing.assert_allclose(vy_f, 0.0, atol=1e-6,
            err_msg="Velocity y-component should be zero")


# ---------------------------------------------------------------------------
# 7. Spatially-varying filter scale
# ---------------------------------------------------------------------------

class TestSpatiallyVaryingScale:
    """Passing an array for k (one scale per node) should work."""

    def test_array_scale_nodes(self, filt_20):
        filt, n2d, _ = filt_20
        np.random.seed(42)
        data = np.random.randn(n2d)
        k_array = np.full(n2d, 5.0)

        result_scalar = filt.compute(1, 5.0, data)
        result_array = filt.compute(1, k_array, data)

        np.testing.assert_allclose(result_array, result_scalar, atol=1e-6,
            err_msg="Uniform-array scale should match scalar scale")

    def test_array_scale_elements(self, filt_20):
        filt, _, e2d = filt_20
        np.random.seed(42)
        data = np.random.randn(e2d)
        k_array = np.full(e2d, 5.0)

        result_scalar = filt.compute(1, 5.0, data)
        result_array = filt.compute(1, k_array, data)

        np.testing.assert_allclose(result_array, result_scalar, atol=1e-6,
            err_msg="Uniform-array scale should match scalar scale for elements")
