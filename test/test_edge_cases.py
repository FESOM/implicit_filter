"""
Edge case tests: minimal meshes, extreme filter parameters, degenerate inputs.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


def build_filter_from_params(n2d, e2d, tri, xcoord, ycoord, filter_elements=True):
    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=False,
        filter_elements=filter_elements,
    )
    return filt


# ---------------------------------------------------------------------------
# Minimal mesh: single triangle (3 nodes, 1 element)
# ---------------------------------------------------------------------------

class TestMinimalMesh:
    """The filter should work even on the smallest possible mesh."""

    def test_single_triangle_nodes(self):
        tri = np.array([[0, 1, 2]])
        xcoord = np.array([0.0, 1.0, 0.0])
        ycoord = np.array([0.0, 0.0, 1.0])
        filt = build_filter_from_params(3, 1, tri, xcoord, ycoord, filter_elements=False)

        data = np.array([1.0, 2.0, 3.0])
        result = filt.compute(1, 1.0, data)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_single_triangle_elements(self):
        tri = np.array([[0, 1, 2]])
        xcoord = np.array([0.0, 1.0, 0.0])
        ycoord = np.array([0.0, 0.0, 1.0])
        filt = build_filter_from_params(3, 1, tri, xcoord, ycoord, filter_elements=True)

        data = np.array([5.0])
        result = filt.compute(1, 1.0, data)
        assert result.shape == (1,)
        # Single element must return the same value (no neighbors)
        np.testing.assert_allclose(result, 5.0, atol=1e-6)

    def test_two_triangles(self):
        tri = np.array([[0, 1, 2], [1, 3, 2]])
        xcoord = np.array([0.0, 1.0, 0.0, 1.0])
        ycoord = np.array([0.0, 0.0, 1.0, 1.0])
        filt = build_filter_from_params(4, 2, tri, xcoord, ycoord)

        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = filt.compute(1, 1.0, data)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Large filter scale → identity behavior (minimal smoothing)
# ---------------------------------------------------------------------------

class TestLargeScale:
    """Very large filter scale means almost no smoothing."""

    def test_large_scale_preserves_field(self):
        Lx = 10
        xx = np.arange(0, Lx + 1, dtype=float)
        nx = len(xx)
        nodnum = np.reshape(np.arange(nx * nx), [nx, nx]).T
        xcoord = np.zeros((nx, nx))
        ycoord = xcoord.copy()
        for i in range(nx):
            ycoord[i, :] = xx
            xcoord[:, i] = xx
        tri = make_tri(nodnum, nx, nx)

        filt = build_filter_from_params(
            nx * nx, len(tri), tri,
            xcoord.flatten(), ycoord.flatten(),
            filter_elements=False,
        )

        np.random.seed(42)
        data = np.random.randn(nx * nx)

        # k = 1000 is much larger than the mesh scale (~10)
        result = filt.compute(1, 1000.0, data)

        # Should be very close to the original
        np.testing.assert_allclose(result, data, atol=1e-2,
            err_msg="Large-scale filter should approximate identity")


# ---------------------------------------------------------------------------
# Very small filter scale → heavy smoothing (variance collapses)
# ---------------------------------------------------------------------------

class TestSmallScale:
    """Very small filter scale means aggressive smoothing."""

    def test_small_scale_reduces_variance(self):
        Lx = 10
        xx = np.arange(0, Lx + 1, dtype=float)
        nx = len(xx)
        nodnum = np.reshape(np.arange(nx * nx), [nx, nx]).T
        xcoord = np.zeros((nx, nx))
        ycoord = xcoord.copy()
        for i in range(nx):
            ycoord[i, :] = xx
            xcoord[:, i] = xx
        tri = make_tri(nodnum, nx, nx)

        filt = build_filter_from_params(
            nx * nx, len(tri), tri,
            xcoord.flatten(), ycoord.flatten(),
            filter_elements=False,
        )

        np.random.seed(42)
        data = np.random.randn(nx * nx)
        original_var = np.var(data)

        # k = 0.5 is smaller than mesh spacing
        result = filt.compute(1, 0.5, data)
        filtered_var = np.var(result)

        # Variance should be much smaller
        assert filtered_var < 0.1 * original_var, \
            f"Small-scale filter didn't reduce variance enough: {filtered_var:.4f} vs {original_var:.4f}"


# ---------------------------------------------------------------------------
# Partially masked mesh
# ---------------------------------------------------------------------------

class TestMaskedMesh:
    """Mesh with some masked (land) elements should still work."""

    def test_partial_mask(self):
        Lx = 10
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

        # Mask out half the elements
        mask = np.ones(e2d, dtype=bool)
        mask[:e2d // 2] = False  # First half is "land"

        filt = TriangularFilter()
        filt.prepare(
            n2d, e2d, tri, xcoord.flatten(), ycoord.flatten(),
            meshtype='m', cartesian=True, full=False, mask=mask,
            filter_elements=True,
        )

        np.random.seed(42)
        data = np.random.randn(n2d)
        result = filt.compute(1, 5.0, data)
        assert result.shape == (n2d,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Degenerate triangle (zero area) doesn't crash
# ---------------------------------------------------------------------------

class TestDegenerateTriangle:
    """Mesh with a degenerate (zero-area) triangle should not crash during prepare."""

    def test_collinear_nodes_prepare_succeeds(self):
        # Triangle 0 is valid, triangle 1 has collinear nodes (zero area)
        tri = np.array([[0, 1, 2], [1, 3, 4]])
        xcoord = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
        ycoord = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # nodes 1,3,4 are collinear

        filt = TriangularFilter()
        # Should not crash during prepare
        filt.prepare(
            5, 2, tri, xcoord, ycoord,
            meshtype='m', cartesian=True, full=False,
            filter_elements=False,
        )
        # The filter state should exist even with degenerate elements
        assert filt._n2d == 5
        assert filt._e2d == 2
