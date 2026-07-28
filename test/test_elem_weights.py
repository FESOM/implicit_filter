"""
Tests for the element-Laplacian weighting scheme.

The historical operator uses a fixed -sqrt(3)/elem_area off-diagonal, which is
the finite-volume coefficient edge_length/centroid_distance specialised to
equilateral triangles. The 'geometric' scheme instead uses the per-edge weight
actually computed from the mesh, so the two must agree on an equilateral mesh
and diverge on an anisotropic one.
"""
import numpy as np
import pytest

from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri


def equilateral_mesh(nx=8, ny=8, s=1.0):
    """A genuine equilateral triangulation (offset rows)."""
    h = s * np.sqrt(3.0) / 2.0
    xcoord, ycoord = [], []
    for j in range(ny):
        for i in range(nx):
            xcoord.append(i * s + (j % 2) * (s / 2.0))
            ycoord.append(j * h)
    xcoord = np.array(xcoord)
    ycoord = np.array(ycoord)

    def nid(j, i):
        return j * nx + i

    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if j % 2 == 0:
                tri.append([nid(j, i), nid(j, i + 1), nid(j + 1, i)])
                tri.append([nid(j, i + 1), nid(j + 1, i + 1), nid(j + 1, i)])
            else:
                tri.append([nid(j, i), nid(j, i + 1), nid(j + 1, i + 1)])
                tri.append([nid(j, i), nid(j + 1, i + 1), nid(j + 1, i)])
    tri = np.array(tri, dtype=int)
    return len(xcoord), len(tri), tri, xcoord, ycoord


def stretched_mesh(nx=8, ny=8, stretch=6.0):
    """Right triangles stretched hard in x, so cells are far from equilateral."""
    nodnum = np.reshape(np.arange(nx * ny), [ny, nx]).T
    xc = np.zeros((nx, ny))
    yc = np.zeros((nx, ny))
    for i in range(nx):
        yc[i, :] = np.arange(ny, dtype=float)
    for i in range(ny):
        xc[:, i] = np.arange(nx, dtype=float) * stretch
    tri = make_tri(nodnum, nx, ny)
    return nx * ny, len(tri), tri, xc.flatten(), yc.flatten()


def build(mesh, **kw):
    n2d, e2d, tri, xcoord, ycoord = mesh
    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype="m", cartesian=True, full=False,
        filter_elements=True, **kw,
    )
    return filt


def elem_operator(filt):
    return np.asarray(filt._ss_e), np.asarray(filt._ii_e), np.asarray(filt._jj_e)


class TestDefaultUnchanged:
    """The default must remain the historical equilateral operator."""

    def test_default_matches_explicit_equilateral(self):
        a = build(stretched_mesh())
        b = build(stretched_mesh(), elem_weights="equilateral")
        for x, y in zip(elem_operator(a), elem_operator(b)):
            np.testing.assert_array_equal(x, y)

    def test_default_operator_is_area_only(self):
        """Historical behaviour: off-diagonals depend only on element area."""
        filt = build(stretched_mesh())
        ss, ii, jj = elem_operator(filt)
        off = ss[ii != jj]
        rows = ii[ii != jj]
        elem_area = np.asarray(filt._elem_area)
        expected = -np.sqrt(3) / elem_area[rows]
        np.testing.assert_allclose(off, expected, rtol=1e-12)


class TestGeometricScheme:
    def test_agrees_with_equilateral_on_equilateral_mesh(self):
        """Both schemes must coincide when the cells really are equilateral."""
        eq = build(equilateral_mesh(), elem_weights="equilateral")
        geo = build(equilateral_mesh(), elem_weights="geometric")

        ss_eq, ii_eq, jj_eq = elem_operator(eq)
        ss_geo, ii_geo, jj_geo = elem_operator(geo)

        np.testing.assert_array_equal(ii_eq, ii_geo)
        np.testing.assert_array_equal(jj_eq, jj_geo)
        # interior faces only: boundary cells have fewer neighbours
        interior = ii_eq != jj_eq
        np.testing.assert_allclose(
            ss_geo[interior], ss_eq[interior], rtol=1e-6,
            err_msg="geometric weights must reduce to sqrt(3) on equilateral cells",
        )

    def test_differs_on_stretched_mesh(self):
        eq = build(stretched_mesh(), elem_weights="equilateral")
        geo = build(stretched_mesh(), elem_weights="geometric")
        ss_eq, _, _ = elem_operator(eq)
        ss_geo, _, _ = elem_operator(geo)
        assert not np.allclose(ss_eq, ss_geo), (
            "on a strongly anisotropic mesh the geometric operator must differ "
            "from the equilateral approximation"
        )

    def test_geometric_responds_to_stretching(self):
        """Increasing anisotropy must change the geometric operator."""
        a, _, _ = elem_operator(build(stretched_mesh(stretch=2.0), elem_weights="geometric"))
        b, _, _ = elem_operator(build(stretched_mesh(stretch=8.0), elem_weights="geometric"))
        assert not np.allclose(a, b)

    def test_rows_sum_to_zero(self):
        """A Laplacian annihilates constants, whichever scheme is used."""
        for scheme in ("equilateral", "geometric"):
            filt = build(stretched_mesh(), elem_weights=scheme)
            ss, ii, _ = elem_operator(filt)
            rows = np.zeros(filt._e2d)
            np.add.at(rows, ii, ss)
            np.testing.assert_allclose(
                rows, 0.0, atol=1e-9, err_msg=f"scheme={scheme}"
            )

    def test_constant_field_preserved(self):
        filt = build(stretched_mesh(), elem_weights="geometric")
        result = filt.compute(1, 2.0, np.full(filt._e2d, 6.0))
        np.testing.assert_allclose(result, 6.0, atol=1e-5)

    def test_variance_decreases(self):
        filt = build(stretched_mesh(), elem_weights="geometric")
        np.random.seed(11)
        data = np.random.randn(filt._e2d)
        result = filt.compute(1, 2.0, data)
        assert np.var(result) <= np.var(data) + 1e-10

    def test_offdiagonals_negative(self):
        filt = build(stretched_mesh(), elem_weights="geometric")
        ss, ii, jj = elem_operator(filt)
        off = ss[ii != jj]
        assert np.all(off < 0), "off-diagonal Laplacian entries must be negative"


class TestValidation:
    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="elem_weights"):
            build(stretched_mesh(), elem_weights="banana")

    def test_node_filtering_unaffected_by_scheme(self):
        """elem_weights must not touch the nodal operator."""
        a = build(stretched_mesh(), elem_weights="equilateral")
        b = build(stretched_mesh(), elem_weights="geometric")
        np.testing.assert_array_equal(np.asarray(a._ss), np.asarray(b._ss))
