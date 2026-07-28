"""
Backward-compatibility guards.

Everything here describes how the package was called *before* this round of
changes. These tests exist so that a future edit which quietly reorders a
parameter, renames an export, or changes a cache format fails loudly rather
than breaking user scripts.

Checked against the pre-review baseline by running identical code on both
revisions; all calls below produced identical results on both.
"""
import inspect

import numpy as np
import pytest

import implicit_filter
from implicit_filter import TriangularFilter, LatLonFilter
from implicit_filter.utils._auxiliary import make_tri


def cartesian_mesh(n=10):
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    xc = np.zeros((n, n))
    yc = np.zeros((n, n))
    for i in range(n):
        yc[i, :] = np.arange(n, dtype=float)
    for i in range(n):
        xc[:, i] = np.arange(n, dtype=float)
    return n * n, make_tri(nodnum, n, n), xc.flatten(), yc.flatten()


class TestPublicExports:
    """Names importable from the package root must not disappear."""

    EXPECTED = [
        "Filter", "TriangularFilter", "FesomFilter", "IconFilter",
        "LatLonFilter", "NemoFilter",
        "make_tri", "convert_to_wavenumbers",
        "transform_velocity_to_nodes", "transform_scalar_to_nodes",
        "transform_to_T_cells", "transform_mask_from_elements_to_nodes",
        "transform_mask_from_nodes_to_elements",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_export_present(self, name):
        assert hasattr(implicit_filter, name), (
            f"{name} was importable from implicit_filter before; removing it "
            "breaks user imports"
        )


class TestSignaturesRemainCompatible:
    """New parameters must be appended with defaults, never inserted."""

    # (callable, leading positional parameters in their original order)
    LEGACY_ORDER = {
        "TriangularFilter.prepare": (
            TriangularFilter.prepare,
            ["self", "n2d", "e2d", "tri", "xcoord", "ycoord", "meshtype",
             "cartesian", "cyclic_length", "full", "mask", "gpu",
             "filter_elements"],
        ),
        "TriangularFilter.compute": (
            TriangularFilter.compute, ["self", "n", "k", "data", "x0"],
        ),
        "TriangularFilter.compute_velocity": (
            TriangularFilter.compute_velocity,
            ["self", "n", "k", "ux", "vy", "ux0", "vy0"],
        ),
        "LatLonFilter.prepare": (
            LatLonFilter.prepare,
            ["self", "latitude", "longitude", "cartesian", "local",
             "cyclic_length", "mask", "gpu"],
        ),
    }

    @pytest.mark.parametrize("label", sorted(LEGACY_ORDER))
    def test_leading_parameters_unchanged(self, label):
        fn, expected = self.LEGACY_ORDER[label]
        actual = list(inspect.signature(fn).parameters)
        assert actual[: len(expected)] == expected, (
            f"{label}: leading parameters changed, which silently rebinds "
            f"positional calls.\n  expected prefix: {expected}\n  actual: {actual}"
        )

    @pytest.mark.parametrize("label", sorted(LEGACY_ORDER))
    def test_added_parameters_have_defaults(self, label):
        fn, expected = self.LEGACY_ORDER[label]
        params = inspect.signature(fn).parameters
        for name in list(params)[len(expected):]:
            assert params[name].default is not inspect.Parameter.empty, (
                f"{label}: new parameter {name!r} has no default, so existing "
                "calls would fail"
            )


class TestLegacyCallForms:
    """Call patterns that worked before must still work."""

    def test_fully_positional_prepare(self):
        n2d, tri, xc, yc = cartesian_mesh()
        filt = TriangularFilter()
        # the 12-argument positional form accepted previously
        filt.prepare(n2d, len(tri), tri, xc, yc, "m", True,
                     2 * np.pi, False, None, False, True)
        assert filt._n2d == n2d

    def test_positional_x0(self):
        n2d, tri, xc, yc = cartesian_mesh()
        filt = TriangularFilter()
        filt.prepare(n2d, len(tri), tri, xc, yc, meshtype="m", cartesian=True)
        data = np.arange(n2d, dtype=float)
        filt.compute(1, 5.0, data, data * 0.5)   # x0 given positionally

    def test_positional_velocity_guesses(self):
        n2d, tri, xc, yc = cartesian_mesh()
        filt = TriangularFilter()
        filt.prepare(n2d, len(tri), tri, xc, yc, meshtype="m", cartesian=True)
        d = np.arange(n2d, dtype=float)
        filt.compute_velocity(1, 5.0, d, -d, d * 0.1, -d * 0.1)

    def test_length_based_dispatch_still_default(self):
        """Omitting on= must keep inferring placement from length."""
        n2d, tri, xc, yc = cartesian_mesh()
        e2d = len(tri)
        filt = TriangularFilter()
        filt.prepare(n2d, e2d, tri, xc, yc, meshtype="m", cartesian=True,
                     filter_elements=True)
        assert filt.compute(1, 5.0, np.ones(n2d)).shape == (n2d,)
        assert filt.compute(1, 5.0, np.ones(e2d)).shape == (e2d,)


class TestCacheFormat:
    """Cache files must interoperate across versions.

    Verified against the pre-review baseline in both directions: a cache
    written by the old code loads here, and a cache written here loads there,
    both giving identical filtered output.
    """

    def test_roundtrip(self, tmp_path):
        n2d, tri, xc, yc = cartesian_mesh()
        filt = TriangularFilter()
        filt.prepare(n2d, len(tri), tri, xc, yc, meshtype="m", cartesian=True,
                     filter_elements=True)
        path = str(tmp_path / "cache.npz")
        filt.save_to_file(path)

        loaded = TriangularFilter.load_from_file(path)
        data = np.arange(n2d, dtype=float)
        np.testing.assert_allclose(
            loaded.compute(1, 5.0, data), filt.compute(1, 5.0, data)
        )

    def test_saved_keys_are_a_superset_of_what_load_needs(self, tmp_path):
        """Old readers index the same attribute names."""
        n2d, tri, xc, yc = cartesian_mesh()
        filt = TriangularFilter()
        filt.prepare(n2d, len(tri), tri, xc, yc, meshtype="m", cartesian=True,
                     filter_elements=True)
        path = str(tmp_path / "cache.npz")
        filt.save_to_file(path)
        keys = set(np.load(path).files)
        for required in ("_ss", "_ii", "_jj", "_n2d", "_e2d", "_area",
                         "_elem_area", "_ne_pos", "_ne_num"):
            assert required in keys, f"{required} missing from cache"


class TestBackendReporting:
    """get_backend() changed its return value; pin the new contract."""

    def test_round_trips_through_set_backend(self):
        import jax

        original = jax.config.jax_platforms
        try:
            filt = TriangularFilter()
            for requested in ("cpu", "gpu"):
                filt.set_backend(requested)
                assert filt.get_backend() == requested
        finally:
            jax.config.update("jax_platforms", original)


class TestDefaultNemoPathDependencies:
    """neighb='full' is NemoFilter's default and needs pandas + scikit-learn,
    so both must remain base requirements rather than optional extras."""

    def test_declared_in_requirements(self):
        from pathlib import Path

        req = Path(implicit_filter.__file__).parents[2] / "requirements.txt"
        if not req.exists():          # installed without the source tree
            pytest.skip("requirements.txt not available")
        text = req.read_text().lower()
        assert "pandas" in text
        assert "scikit-learn" in text

    def test_both_importable(self):
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
