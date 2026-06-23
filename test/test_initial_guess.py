"""
Tests for the new initial guess (x0, ux0, vy0) functionality in the JAX-based solver.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter, LatLonFilter
from implicit_filter.utils._auxiliary import make_tri


def build_filter(filter_elements=False, full=False, n=10):
    """Build a simple prepared TriangularFilter on a regular mesh."""
    xx = np.arange(0, n + 1, dtype=float)
    yy = np.arange(0, n + 1, dtype=float)
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

    mask = np.ones(e2d, dtype=bool)

    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=full,
        mask=mask, filter_elements=filter_elements,
    )
    return filt, n2d, e2d


class TestInitialGuess:
    def test_compute_scalar_with_x0(self):
        """Test compute() with an initial guess (x0)."""
        filt, n2d, _ = build_filter()
        np.random.seed(42)
        data = np.random.randn(n2d)
        
        # Compute without x0
        result_no_x0 = filt.compute(1, 5.0, data)
        
        # Compute with a random x0
        x0_rand = np.random.randn(n2d)
        result_rand_x0 = filt.compute(1, 5.0, data, x0=x0_rand)
        
        # Compute with perfect x0
        result_perfect_x0 = filt.compute(1, 5.0, data, x0=result_no_x0)
        
        # All should give the same result within solver tolerance
        np.testing.assert_allclose(result_no_x0, result_rand_x0, atol=1e-4)
        np.testing.assert_allclose(result_no_x0, result_perfect_x0, atol=1e-4)

    def test_compute_velocity_with_x0(self):
        """Test compute_velocity() with initial guesses (ux0, vy0)."""
        filt, n2d, _ = build_filter(full=False)
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        
        # Compute without x0
        ux_no, vy_no = filt.compute_velocity(1, 5.0, ux, vy)
        
        # Compute with x0
        ux0_rand = np.random.randn(n2d)
        vy0_rand = np.random.randn(n2d)
        ux_yes, vy_yes = filt.compute_velocity(1, 5.0, ux, vy, ux0=ux0_rand, vy0=vy0_rand)
        
        # Compute with perfect x0
        ux_perf, vy_perf = filt.compute_velocity(1, 5.0, ux, vy, ux0=ux_no, vy0=vy_no)
        
        # All should give the same result
        np.testing.assert_allclose(ux_no, ux_yes, atol=1e-4)
        np.testing.assert_allclose(vy_no, vy_yes, atol=1e-4)
        np.testing.assert_allclose(ux_no, ux_perf, atol=1e-4)
        np.testing.assert_allclose(vy_no, vy_perf, atol=1e-4)

    def test_compute_full_with_x0(self):
        """Test coupled metric computation with x0."""
        filt, n2d, _ = build_filter(full=True)
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)
        
        # Compute without x0
        ux_no, vy_no = filt.compute_velocity(1, 5.0, ux, vy)
        
        # Compute with perfect x0
        ux_perf, vy_perf = filt.compute_velocity(1, 5.0, ux, vy, ux0=ux_no, vy0=vy_no)
        
        np.testing.assert_allclose(ux_no, ux_perf, atol=1e-4)
        np.testing.assert_allclose(vy_no, vy_perf, atol=1e-4)

    def test_element_filtering_with_x0(self):
        """Test element-based compute with x0."""
        filt, _, e2d = build_filter(filter_elements=True)
        np.random.seed(42)
        data = np.random.randn(e2d)
        
        result_no_x0 = filt.compute(1, 5.0, data)
        result_perfect_x0 = filt.compute(1, 5.0, data, x0=result_no_x0)
        
        np.testing.assert_allclose(result_no_x0, result_perfect_x0, atol=1e-4)

    def test_invalid_x0_shape(self):
        """Test providing an x0 with an invalid shape."""
        filt, n2d, _ = build_filter()
        data = np.random.randn(n2d)
        
        # x0 is the wrong size (e.g., e2d size or just wrong)
        wrong_x0 = np.random.randn(n2d + 1)
        
        # JAX shape mismatch should raise a ValueError
        with pytest.raises((ValueError, TypeError)):
            filt.compute(1, 5.0, data, x0=wrong_x0)

    def test_latlon_filtering_with_x0(self):
        """Test latlon filtering with x0."""
        filt = LatLonFilter()
        lats = np.linspace(-80, 80, 10)
        lons = np.linspace(-180, 180, 20)
        filt.prepare(lats, lons)
        
        np.random.seed(42)
        data = np.random.randn(20, 10)
        
        result_no_x0 = filt.compute(1, 5.0, data)
        result_perfect_x0 = filt.compute(1, 5.0, data, x0=result_no_x0)
        
        np.testing.assert_allclose(result_no_x0, result_perfect_x0, atol=1e-4)
