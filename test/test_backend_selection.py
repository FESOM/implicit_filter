"""
set_backend rejects unknown backends (as it did before the JAX migration), and
the deprecated gpu= argument forwards to it instead of doing nothing.
"""
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from implicit_filter import LatLonFilter, TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


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


def tri_args(n=6):
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    xc = np.zeros((n, n))
    yc = np.zeros((n, n))
    for i in range(n):
        yc[i, :] = np.arange(n, dtype=float)
    for i in range(n):
        xc[:, i] = np.arange(n, dtype=float)
    tri = make_tri(nodnum, n, n)
    return n * n, len(tri), tri, xc.flatten(), yc.flatten()


class TestSetBackendValidates:
    @pytest.mark.parametrize("bad", ["quantum", "", "cuda", "gpu0", "cpu gpu", "none"])
    def test_unknown_backend_raises(self, bad):
        filt = TriangularFilter()
        with pytest.raises(NotImplementedError, match="is not supported"):
            filt.set_backend(bad)

    def test_message_matches_the_pre_migration_wording(self):
        filt = TriangularFilter()
        with pytest.raises(NotImplementedError) as exc:
            filt.set_backend("quantum")
        assert str(exc.value) == "Backend quantum is not supported."

    def test_rejection_leaves_the_platform_untouched(self):
        filt = TriangularFilter()
        filt.set_backend("cpu")
        before = jax.config.jax_platforms
        with pytest.raises(NotImplementedError):
            filt.set_backend("quantum")
        assert jax.config.jax_platforms == before

    @pytest.mark.parametrize("good, expected", [
        ("cpu", "cpu"), ("gpu", "gpu"), ("CPU", "cpu"), ("GPU", "gpu"), (" cpu ", "cpu")])
    def test_accepted_backends(self, good, expected):
        filt = TriangularFilter()
        filt.set_backend(good)
        assert filt.get_backend() == expected

    def test_non_string_raises(self):
        filt = TriangularFilter()
        with pytest.raises(NotImplementedError):
            filt.set_backend(None)


class TestDeprecatedGpuArgument:
    """gpu=True must warn *and* select the GPU backend, as it did before."""

    def test_triangular_prepare_gpu_true_selects_gpu(self):
        n2d, e2d, tri, x, y = tri_args()
        filt = TriangularFilter()
        filt.set_backend("cpu")
        with pytest.warns(DeprecationWarning, match="set_backend"):
            filt.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True, gpu=True)
        assert filt.get_backend() == "gpu"

    def test_latlon_prepare_gpu_true_selects_gpu(self):
        filt = LatLonFilter()
        filt.set_backend("cpu")
        with pytest.warns(DeprecationWarning, match="set_backend"):
            filt.prepare(np.linspace(-5.0, 5.0, 8), np.linspace(0.0, 10.0, 9),
                         cartesian=True, local=True, gpu=True)
        assert filt.get_backend() == "gpu"

    def test_default_is_silent_and_keeps_the_selected_backend(self):
        n2d, e2d, tri, x, y = tri_args()
        filt = TriangularFilter()
        filt.set_backend("gpu")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            filt.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True)
        assert filt.get_backend() == "gpu", (
            "gpu=False is the default and must not reset a backend the user "
            "selected, unlike the pre-migration unconditional set_backend"
        )

    def test_rejected_prepare_leaves_the_platform_untouched(self):
        """The forwarding runs after prepare's own argument validation."""
        n2d, e2d, tri, x, y = tri_args()
        filt = TriangularFilter()
        filt.set_backend("cpu")
        before = jax.config.jax_platforms
        with pytest.raises(ValueError):
            filt.prepare(n2d, e2d, tri, x, y, meshtype="bogus", cartesian=True,
                         gpu=True)
        assert jax.config.jax_platforms == before
        assert filt.get_backend() == "cpu"

    def test_warning_does_not_claim_the_argument_is_inert(self):
        n2d, e2d, tri, x, y = tri_args()
        filt = TriangularFilter()
        with pytest.warns(DeprecationWarning) as rec:
            filt.prepare(n2d, e2d, tri, x, y, meshtype="m", cartesian=True, gpu=True)
        assert "never had an effect" not in str(rec[0].message)
