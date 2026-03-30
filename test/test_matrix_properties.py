"""
Tests for mathematical properties of assembled sparse matrices.
The Laplacian must satisfy: symmetric, zero row-sums, non-positive off-diagonal,
non-negative diagonal, and positive semi-definiteness.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import (
    make_tri, neighboring_triangles, neighbouring_nodes, areas,
    find_and_sort_edges_and_triangles, calculate_triangle_centers,
)
from implicit_filter.utils._jax_elem_function import (
    vectorized_orient_edges,
    vectorized_calculate_dimensional_quantities,
    fast_calculate_laplacian_weights,
    fast_build_smoothing_and_metric,
    fast_assemble_from_intermediate,
)
from implicit_filter.utils._jax_function import make_smooth, make_smat
import jax.numpy as jnp


def build_mesh(Lx=10):
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
    xcoord = xcoord.flatten()
    ycoord = ycoord.flatten()
    tri = make_tri(nodnum, nx, ny)
    n2d = len(xcoord)
    e2d = len(tri)
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    mask = np.ones(e2d)
    area, elem_area, dx, dy, Mt = areas(
        n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
        "m", True, 0, mask,
    )
    return {
        'n2d': n2d, 'e2d': e2d, 'tri': tri,
        'xcoord': xcoord, 'ycoord': ycoord,
        'nn_num': nn_num, 'nn_pos': nn_pos,
        'ne_num': ne_num, 'ne_pos': ne_pos,
        'elem_area': elem_area, 'Mt': Mt, 'area': area,
        'dx': dx, 'dy': dy,
    }


def build_nodal_laplacian(m):
    """Build the nodal Laplacian as a dense matrix.
    Returns both raw S (symmetric) and area-weighted D^{-1}S (used for filtering).
    """
    smooth, _ = make_smooth(
        jnp.array(m['Mt']), jnp.array(m['elem_area']),
        jnp.array(m['dx']), jnp.array(m['dy']),
        jnp.array(m['nn_num']), jnp.array(m['nn_pos']),
        jnp.array(m['tri']), m['n2d'], m['e2d'], full=False,
    )
    # Raw (symmetric) smoothness matrix
    nza = int(jnp.sum(jnp.array(m['nn_num'])))
    ss_raw, ii_raw, jj_raw = make_smat(
        jnp.array(m['nn_pos']), jnp.array(m['nn_num']),
        smooth, m['n2d'], nza,
    )
    S_raw = coo_matrix(
        (np.array(ss_raw), (np.array(ii_raw), np.array(jj_raw))),
        shape=(m['n2d'], m['n2d'])
    ).toarray()

    # Area-weighted (used for filtering): D^{-1} S
    area = jnp.array(m['area'])
    for n in range(m['n2d']):
        smooth = smooth.at[:, n].set(smooth[:, n] / area[n])
    ss, ii, jj = make_smat(
        jnp.array(m['nn_pos']), jnp.array(m['nn_num']),
        smooth, m['n2d'], nza,
    )
    S_weighted = coo_matrix(
        (np.array(ss), (np.array(ii), np.array(jj))),
        shape=(m['n2d'], m['n2d'])
    ).toarray()
    return S_raw, S_weighted


def build_element_laplacian(m):
    """Build the element Laplacian as a dense matrix."""
    edges, edge_tri, ed2d_in = find_and_sort_edges_and_triangles(
        m['n2d'], m['nn_num'], m['nn_pos'], m['ne_num'], m['ne_pos']
    )
    tcenter = calculate_triangle_centers(
        m['e2d'], m['tri'], m['xcoord'], m['ycoord'], 'm', 0
    )
    edges, edge_tri = vectorized_orient_edges(
        edges.shape[1], edges, edge_tri, tcenter, m['xcoord'], m['ycoord']
    )
    dxdy, cross = vectorized_calculate_dimensional_quantities(
        edges.shape[1], ed2d_in, edges, edge_tri, tcenter, m['xcoord'], m['ycoord']
    )
    ee_pos, ee_num, _, dxcell = fast_calculate_laplacian_weights(
        m['e2d'], ed2d_in, edge_tri, dxdy, cross
    )
    sm, _ = fast_build_smoothing_and_metric(
        m['e2d'], m['n2d'], ee_num, ee_pos, m['elem_area'], False
    )
    ss, ii, jj = fast_assemble_from_intermediate(m['e2d'], ee_num, ee_pos, sm)
    S = coo_matrix(
        (ss, (ii.astype(int), jj.astype(int))),
        shape=(m['e2d'], m['e2d'])
    ).toarray()
    return S


@pytest.fixture
def nodal_matrix():
    m = build_mesh(10)
    S_raw, S_weighted = build_nodal_laplacian(m)
    return S_raw, S_weighted, m['n2d']


@pytest.fixture
def element_matrix():
    m = build_mesh(10)
    return build_element_laplacian(m), m['e2d']


# ---------------------------------------------------------------------------
# Row sums = 0  (Laplacian of a constant is zero)
# ---------------------------------------------------------------------------

class TestRowSums:
    def test_nodal_row_sums_zero(self, nodal_matrix):
        S_raw, S_weighted, n = nodal_matrix
        # Raw S has zero row sums
        row_sums = S_raw.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10,
            err_msg="Raw nodal Laplacian row sums are not zero")

    def test_element_row_sums_zero(self, element_matrix):
        S, e = element_matrix
        row_sums = S.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10,
            err_msg="Element Laplacian row sums are not zero")


# ---------------------------------------------------------------------------
# Symmetry  (S == S.T)
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_nodal_raw_symmetry(self, nodal_matrix):
        S_raw, _, _ = nodal_matrix
        np.testing.assert_allclose(S_raw, S_raw.T, atol=1e-10,
            err_msg="Raw nodal Laplacian is not symmetric")

    def test_element_symmetry(self, element_matrix):
        S, _ = element_matrix
        np.testing.assert_allclose(S, S.T, atol=1e-10,
            err_msg="Element Laplacian is not symmetric")


# ---------------------------------------------------------------------------
# Sign structure: off-diagonal ≤ 0, diagonal ≥ 0
# ---------------------------------------------------------------------------

class TestSignStructure:
    def test_nodal_diagonal_nonnegative(self, nodal_matrix):
        S_raw, _, _ = nodal_matrix
        assert np.all(np.diag(S_raw) >= -1e-12), \
            "Nodal Laplacian has negative diagonal entries"

    def test_nodal_offdiag_nonpositive(self, nodal_matrix):
        S_raw, _, n = nodal_matrix
        off_diag = S_raw - np.diag(np.diag(S_raw))
        assert np.all(off_diag <= 1e-12), \
            "Nodal Laplacian has positive off-diagonal entries"

    def test_element_diagonal_nonnegative(self, element_matrix):
        S, _ = element_matrix
        assert np.all(np.diag(S) >= -1e-12), \
            "Element Laplacian has negative diagonal entries"

    def test_element_offdiag_nonpositive(self, element_matrix):
        S, _ = element_matrix
        off_diag = S - np.diag(np.diag(S))
        assert np.all(off_diag <= 1e-12), \
            "Element Laplacian has positive off-diagonal entries"


# ---------------------------------------------------------------------------
# Positive semi-definiteness (all eigenvalues ≥ 0)
# ---------------------------------------------------------------------------

class TestPositiveSemiDefinite:
    def test_nodal_raw_psd(self, nodal_matrix):
        S_raw, _, _ = nodal_matrix
        eigvals = np.linalg.eigvalsh(S_raw)
        assert np.all(eigvals >= -1e-8), \
            f"Raw nodal Laplacian has negative eigenvalue: {eigvals.min():.2e}"

    def test_element_psd(self, element_matrix):
        S, _ = element_matrix
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-8), \
            f"Element Laplacian has negative eigenvalue: {eigvals.min():.2e}"
