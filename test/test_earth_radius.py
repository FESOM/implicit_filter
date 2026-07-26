"""
The Earth radius must be one shared constant.

It was previously hardcoded three times: 6400 in _auxiliary.areas (the nodal
geometry), 6400 in latlon_filter, and 6371 in the element branch of
TriangularFilter.prepare -- so a single prepared filter mixed two radii.

The element-path value turns out not to reach the assembled operator today
(the per-edge weights are scale-invariant and the metric term is discarded),
which is why unifying it is a no-op. That invariance is pinned below so the
inconsistency cannot silently come back to life.
"""
import re
from pathlib import Path

import numpy as np
import pytest

import implicit_filter
from implicit_filter.utils._auxiliary import R_EARTH, make_tri
from implicit_filter import TriangularFilter


SRC = Path(implicit_filter.__file__).parent


def spherical_mesh(n=12):
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    for i in range(n):
        lat[i, :] = np.linspace(30.0, 40.0, n)
    for i in range(n):
        lon[:, i] = np.linspace(0.0, 10.0, n)
    tri = make_tri(nodnum, n, n)
    return n * n, len(tri), tri, lon.flatten(), lat.flatten()


class TestSharedConstant:
    def test_constant_exists_and_is_sane(self):
        assert 6300.0 < R_EARTH < 6500.0, "Earth radius should be in km"

    def test_no_stray_radius_literals_in_source(self):
        """No module may hardcode its own Earth radius."""
        offenders = []
        for path in SRC.rglob("*.py"):
            if ".ipynb_checkpoints" in str(path):
                continue
            text = path.read_text()
            for lineno, line in enumerate(text.splitlines(), 1):
                if re.match(r"\s*R_EARTH\s*=", line):
                    continue  # the canonical definition itself
                if re.search(r"r_earth\s*=\s*6[0-9]{3}", line, re.IGNORECASE):
                    offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
        assert not offenders, (
            "Earth radius must come from _auxiliary.R_EARTH, found literals:\n"
            + "\n".join(offenders)
        )

    def test_nodal_geometry_uses_the_constant(self):
        """Scaling R_EARTH must scale the nodal element areas as R^2."""
        from implicit_filter.utils import _auxiliary as aux

        n2d, e2d, tri, xcoord, ycoord = spherical_mesh()
        ne_num, ne_pos = aux.neighboring_triangles(n2d, e2d, tri)
        mask = np.ones(e2d)

        def areas_with(radius):
            original = aux.R_EARTH
            aux.R_EARTH = radius
            try:
                return aux.areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
                                 "r", False, 2 * np.pi, mask)[1]
            finally:
                aux.R_EARTH = original

        base = areas_with(R_EARTH)
        doubled = areas_with(2.0 * R_EARTH)
        np.testing.assert_allclose(doubled, 4.0 * base, rtol=1e-10)


class TestOneRadiusThroughout:
    """A single radius must govern the whole filter, nodal and element alike.

    The element operator's off-diagonals go as 1/elem_area and elem_area goes
    as R^2, so scaling R must scale the operator by exactly 1/R^2. Before the
    fix the element branch used 6371 while the areas used 6400, and no single
    scaling could describe the result.
    """

    def _element_operator(self, radius, scheme):
        from implicit_filter.utils import _auxiliary as aux

        original = aux.R_EARTH
        aux.R_EARTH = radius
        try:
            n2d, e2d, tri, xcoord, ycoord = spherical_mesh()
            filt = TriangularFilter()
            filt.prepare(n2d, e2d, tri, xcoord, ycoord, meshtype="r",
                         cartesian=False, filter_elements=True,
                         elem_weights=scheme)
            return np.asarray(filt._ss_e)
        finally:
            aux.R_EARTH = original

    @pytest.mark.parametrize("scheme", ["equilateral", "geometric"])
    def test_element_operator_scales_as_inverse_r_squared(self, scheme):
        base = self._element_operator(R_EARTH, scheme)
        doubled = self._element_operator(2.0 * R_EARTH, scheme)
        np.testing.assert_allclose(doubled, base / 4.0, rtol=1e-9)

    def test_nodal_operator_scales_as_inverse_r_squared(self):
        from implicit_filter.utils import _auxiliary as aux

        def operator_with(radius):
            original = aux.R_EARTH
            aux.R_EARTH = radius
            try:
                n2d, e2d, tri, xcoord, ycoord = spherical_mesh()
                filt = TriangularFilter()
                filt.prepare(n2d, e2d, tri, xcoord, ycoord, meshtype="r",
                             cartesian=False)
                return np.asarray(filt._ss)
            finally:
                aux.R_EARTH = original

        np.testing.assert_allclose(
            operator_with(2.0 * R_EARTH), operator_with(R_EARTH) / 4.0, rtol=1e-5
        )
