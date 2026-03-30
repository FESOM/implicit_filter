"""
Serialization tests: save_to_file / load_from_file round-trip.
"""
import os
import numpy as np
import pytest
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


def build_filter():
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
        meshtype='m', cartesian=True, full=False,
        filter_elements=True,
    )
    return filt, n2d, e2d


@pytest.fixture
def save_path(tmp_path):
    return str(tmp_path / "filter_state.npz")


class TestSaveLoad:
    def test_file_created(self, save_path):
        filt, _, _ = build_filter()
        filt.save_to_file(save_path)
        assert os.path.isfile(save_path), "Save file was not created"

    def test_file_is_valid_npz(self, save_path):
        filt, _, _ = build_filter()
        filt.save_to_file(save_path)
        data = np.load(save_path)
        assert len(data.files) > 0, "NPZ file has no arrays"

    def test_roundtrip_node_compute(self, save_path):
        filt, n2d, _ = build_filter()
        np.random.seed(42)
        data = np.random.randn(n2d)

        result_before = filt.compute(1, 5.0, data)

        filt.save_to_file(save_path)
        loaded = TriangularFilter.load_from_file(save_path)
        result_after = loaded.compute(1, 5.0, data)

        np.testing.assert_allclose(result_after, result_before, atol=1e-10,
            err_msg="Loaded filter gives different result for node data")

    def test_roundtrip_element_compute(self, save_path):
        filt, _, e2d = build_filter()
        np.random.seed(42)
        data = np.random.randn(e2d)

        result_before = filt.compute(1, 5.0, data)

        filt.save_to_file(save_path)
        loaded = TriangularFilter.load_from_file(save_path)
        result_after = loaded.compute(1, 5.0, data)

        np.testing.assert_allclose(result_after, result_before, atol=1e-10,
            err_msg="Loaded filter gives different result for element data")

    def test_roundtrip_velocity(self, save_path):
        filt, n2d, _ = build_filter()
        np.random.seed(42)
        ux = np.random.randn(n2d)
        vy = np.random.randn(n2d)

        ux_before, vy_before = filt.compute_velocity(1, 5.0, ux, vy)

        filt.save_to_file(save_path)
        loaded = TriangularFilter.load_from_file(save_path)
        ux_after, vy_after = loaded.compute_velocity(1, 5.0, ux, vy)

        np.testing.assert_allclose(ux_after, ux_before, atol=1e-10)
        np.testing.assert_allclose(vy_after, vy_before, atol=1e-10)
