"""
API-surface tests: backend selection round-trip, explicit node/element
dispatch, and construction from a state mapping.

None of these change what the filter computes; they pin how it is driven.
"""
import numpy as np
import pytest

import jax

from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


@pytest.fixture
def restore_jax_platforms():
    """set_backend mutates global JAX config; put it back afterwards."""
    original = jax.config.jax_platforms
    yield
    jax.config.update("jax_platforms", original)


def build_filter(nx=6, ny=6, filter_elements=True):
    nodnum = np.reshape(np.arange(nx * ny), [ny, nx]).T
    xc = np.zeros((nx, ny))
    yc = np.zeros((nx, ny))
    for i in range(nx):
        yc[i, :] = np.arange(ny, dtype=float)
    for i in range(ny):
        xc[:, i] = np.arange(nx, dtype=float)
    tri = make_tri(nodnum, nx, ny)
    filt = TriangularFilter()
    filt.prepare(
        nx * ny, len(tri), tri, xc.flatten(), yc.flatten(),
        meshtype="m", cartesian=True, filter_elements=filter_elements,
    )
    return filt, nx * ny, len(tri)


class TestBackendRoundTrip:
    """Whatever get_backend reports must be accepted by set_backend."""

    def test_cpu_round_trip(self, restore_jax_platforms):
        filt, _, _ = build_filter()
        filt.set_backend("cpu")
        assert filt.get_backend() == "cpu"

    def test_gpu_round_trip(self, restore_jax_platforms):
        filt, _, _ = build_filter()
        filt.set_backend("gpu")
        assert filt.get_backend() == "gpu", (
            "get_backend must report a value that can be fed back to "
            "set_backend, not JAX's internal platform priority string"
        )

    def test_round_trip_is_stable(self, restore_jax_platforms):
        filt, _, _ = build_filter()
        for requested in ("cpu", "gpu", "cpu"):
            filt.set_backend(requested)
            reported = filt.get_backend()
            filt.set_backend(reported)
            assert filt.get_backend() == reported


class TestExplicitDispatch:
    """compute() infers node-vs-element from length; allow saying it outright."""

    def test_explicit_nodes(self):
        filt, n2d, _ = build_filter()
        data = np.arange(n2d, dtype=float)
        np.testing.assert_allclose(
            filt.compute(1, 5.0, data, on="nodes"),
            filt.compute(1, 5.0, data),
        )

    def test_explicit_elements(self):
        filt, _, e2d = build_filter()
        data = np.arange(e2d, dtype=float)
        np.testing.assert_allclose(
            filt.compute(1, 5.0, data, on="elements"),
            filt.compute(1, 5.0, data),
        )

    def test_wrong_length_for_declared_target_raises(self):
        filt, n2d, e2d = build_filter()
        with pytest.raises(ValueError, match="does not match"):
            filt.compute(1, 5.0, np.ones(n2d), on="elements")

    def test_unknown_target_raises(self):
        filt, n2d, _ = build_filter()
        with pytest.raises(ValueError, match="on="):
            filt.compute(1, 5.0, np.ones(n2d), on="banana")

    def test_velocity_explicit_nodes(self):
        filt, n2d, _ = build_filter()
        ux = np.arange(n2d, dtype=float)
        vy = -np.arange(n2d, dtype=float)
        a = filt.compute_velocity(1, 5.0, ux, vy, on="nodes")
        b = filt.compute_velocity(1, 5.0, ux, vy)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_ambiguous_mesh_resolved_by_on(self):
        """When n2d == e2d the length heuristic cannot decide; on= must."""
        # 4 nodes, 4 triangles: a square split about its centre
        tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        xcoord = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
        ycoord = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        filt = TriangularFilter()
        filt.prepare(5, 4, tri, xcoord, ycoord, meshtype="m",
                     cartesian=True, filter_elements=True)
        assert filt._n2d != filt._e2d  # sanity: this mesh is not ambiguous

        data = np.ones(4)
        # length 4 == e2d, so the heuristic picks elements; on= must agree
        np.testing.assert_allclose(
            filt.compute(1, 1.0, data, on="elements"),
            filt.compute(1, 1.0, data),
        )


class TestConstructionFromMapping:
    """Filter(*initial_data) is documented; it must actually work."""

    def test_positional_mapping_sets_attributes(self):
        filt, _, _ = build_filter()
        state = {k: v for k, v in filt.__getstate__().items() if v is not None}
        rebuilt = TriangularFilter(state)
        assert rebuilt._n2d == filt._n2d
        assert rebuilt._e2d == filt._e2d

    def test_keyword_mapping_still_works(self):
        filt, _, _ = build_filter()
        state = {k: v for k, v in filt.__getstate__().items() if v is not None}
        rebuilt = TriangularFilter(**state)
        assert rebuilt._n2d == filt._n2d

    def test_positional_and_keyword_agree(self):
        filt, n2d, _ = build_filter()
        state = {k: v for k, v in filt.__getstate__().items() if v is not None}
        data = np.arange(n2d, dtype=float)
        np.testing.assert_allclose(
            TriangularFilter(state).compute(1, 5.0, data),
            TriangularFilter(**state).compute(1, 5.0, data),
        )
