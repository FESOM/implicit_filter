"""
Error handling tests: verify that invalid inputs raise appropriate exceptions.
"""
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri
from implicit_filter.utils.utils import get_backend


def build_filter(filter_elements=False, full=False):
    Lx = 10
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
    tri = make_tri(nodnum, nx, ny)
    n2d = len(xcoord.flatten())
    e2d = len(tri)

    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord.flatten(), ycoord.flatten(),
        meshtype='m', cartesian=True, full=full,
        filter_elements=filter_elements,
    )
    return filt, n2d, e2d


class TestInvalidFilterOrder:
    """Filter order n must be ≥ 1."""

    def test_compute_order_zero(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError, match="positive"):
            filt.compute(0, 5.0, np.ones(n2d))

    def test_compute_order_negative(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError, match="positive"):
            filt.compute(-1, 5.0, np.ones(n2d))

    def test_velocity_order_zero(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError, match="positive"):
            filt.compute_velocity(0, 5.0, np.ones(n2d), np.ones(n2d))


class TestElementFilteringNotPrepared:
    """Element-sized data must fail if filter_elements was False."""

    def test_compute_elements_without_prepare(self):
        filt, _, e2d = build_filter(filter_elements=False)
        with pytest.raises(ValueError, match="filter_elements"):
            filt.compute(1, 5.0, np.ones(e2d))

    def test_velocity_elements_without_prepare(self):
        filt, _, e2d = build_filter(filter_elements=False)
        with pytest.raises(ValueError, match="filter_elements"):
            filt.compute_velocity(1, 5.0, np.ones(e2d), np.ones(e2d))


class TestFullMetricWithElements:
    """Full metric form is not supported for element filtering."""

    def test_compute_full_elements_raises(self):
        filt, _, e2d = build_filter(filter_elements=True, full=True)
        with pytest.raises(ValueError, match="[Ff]ull"):
            filt.compute(1, 5.0, np.ones(e2d))

    def test_velocity_full_elements_raises(self):
        filt, _, e2d = build_filter(filter_elements=True, full=True)
        with pytest.raises(ValueError, match="[Ff]ull"):
            filt.compute_velocity(1, 5.0, np.ones(e2d), np.ones(e2d))


class TestInvalidBackend:
    """Requesting an unsupported backend must raise."""

    def test_invalid_backend_string(self):
        with pytest.raises(NotImplementedError):
            get_backend("tpu")

    def test_invalid_backend_on_filter(self):
        filt, _, _ = build_filter()
        with pytest.raises(NotImplementedError):
            filt.set_backend("quantum")
