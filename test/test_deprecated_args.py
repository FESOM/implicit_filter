"""The deprecated gpu= argument on prepare* must warn and select the GPU."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from implicit_filter import LatLonFilter, TriangularFilter
from test_vcycle_core import structured_tri_mesh


@pytest.fixture(autouse=True)
def restore_jax_platforms():
    """set_backend mutates global JAX config; put it back afterwards.

    Restoring the config string is not enough on its own: JAX initialises a
    backend on the first array created in the process and never releases it,
    so a test that selects the GPU before anything else has touched JAX would
    pin the rest of the session to that backend whatever the string says. Commit
    the process to the CPU here, before the test runs, so that a selection made
    inside a test only ever changes the configuration string.
    """
    original = jax.config.jax_platforms
    jax.config.update("jax_platforms", "cpu")
    jnp.zeros(1).block_until_ready()
    yield
    jax.config.update("jax_platforms", original)


def _tri_args():
    x, y, tri = structured_tri_mesh(6, 6, 10.0)
    return len(x), len(tri), tri, x, y


def test_triangular_prepare_gpu_true_warns():
    n2d, e2d, tri, x, y = _tri_args()
    f = TriangularFilter()
    f.set_backend("cpu")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        f.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True, gpu=True)
    assert f.get_backend() == "gpu"


def test_triangular_prepare_default_is_silent():
    import warnings
    n2d, e2d, tri, x, y = _tri_args()
    f = TriangularFilter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        f.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True)


def test_latlon_prepare_gpu_true_warns():
    f = LatLonFilter()
    f.set_backend("cpu")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        f.prepare(np.linspace(-5.0, 5.0, 8), np.linspace(0.0, 10.0, 9),
                  cartesian=True, local=True, gpu=True)
    assert f.get_backend() == "gpu"
