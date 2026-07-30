"""The gpu= argument on prepare* never had an effect; using it must warn."""
import numpy as np
import pytest

from implicit_filter import LatLonFilter, TriangularFilter
from test_vcycle_core import structured_tri_mesh


def _tri_args():
    x, y, tri = structured_tri_mesh(6, 6, 10.0)
    return len(x), len(tri), tri, x, y


def test_triangular_prepare_gpu_true_warns():
    n2d, e2d, tri, x, y = _tri_args()
    f = TriangularFilter()
    with pytest.warns(DeprecationWarning, match="never had an effect"):
        f.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True, gpu=True)


def test_triangular_prepare_default_is_silent():
    import warnings
    n2d, e2d, tri, x, y = _tri_args()
    f = TriangularFilter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        f.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True)


def test_latlon_prepare_gpu_true_warns():
    f = LatLonFilter()
    with pytest.warns(DeprecationWarning, match="never had an effect"):
        f.prepare(np.linspace(-5.0, 5.0, 8), np.linspace(0.0, 10.0, 9),
                  cartesian=True, local=True, gpu=True)
