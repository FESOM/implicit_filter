"""
Tests that verify the fast/vectorized element filtering functions produce
identical results to the original pure-Python implementations in _auxiliary.py.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from implicit_filter.utils._auxiliary import (
    make_tri,
    neighboring_triangles,
    neighbouring_nodes,
    areas,
    find_and_sort_edges_and_triangles,
    calculate_triangle_centers,
    orient_edges,
    calculate_dimensional_quantities,
    calculate_laplacian_weights,
    build_smoothing_and_metric,
    assemble_from_intermediate,
)
from implicit_filter.utils._jax_elem_function import (
    vectorized_orient_edges,
    vectorized_calculate_dimensional_quantities,
    fast_calculate_laplacian_weights,
    fast_build_smoothing_and_metric,
    fast_assemble_from_intermediate,
)


def build_mesh(Lx=20, dxm=1):
    """Build a triangular mesh and all topology/geometry needed for element filtering."""
    Ly = Lx
    xx = np.arange(0, Lx + 1, dxm)
    yy = np.arange(0, Ly + 1, dxm)
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
    e2d = len(tri[:, 0])

    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    mask = np.ones(e2d)
    area, elem_area, dx, dy, Mt = areas(
        n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
        meshtype="m", carthesian=True, cyclic_length=0, mask=mask,
    )

    edges, edge_tri, ed2d_in = find_and_sort_edges_and_triangles(
        n2d, nn_num, nn_pos, ne_num, ne_pos
    )
    tcenter = calculate_triangle_centers(e2d, tri, xcoord, ycoord, meshtype='m', cyclic_length=0)

    return {
        'n2d': n2d, 'e2d': e2d, 'tri': tri,
        'xcoord': xcoord, 'ycoord': ycoord,
        'nn_num': nn_num, 'nn_pos': nn_pos,
        'ne_num': ne_num, 'ne_pos': ne_pos,
        'elem_area': elem_area, 'Mt': Mt,
        'edges': edges, 'edge_tri': edge_tri, 'ed2d_in': ed2d_in,
        'tcenter': tcenter, 'dx': dx, 'dy': dy,
    }


MESH_SIZES = [(10, 1), (20, 1), (50, 2)]


# =============================================================================
# orient_edges
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_orient_edges_consistency(Lx, dxm):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    edges_ref, etri_ref = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )
    edges_fast, etri_fast = vectorized_orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord']
    )

    assert_array_equal(edges_fast, edges_ref,
        err_msg=f"orient_edges mismatch for Lx={Lx}")
    assert_array_equal(etri_fast, etri_ref,
        err_msg=f"orient_edges edge_tri mismatch for Lx={Lx}")


# =============================================================================
# calculate_dimensional_quantities
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_dimensional_quantities_consistency(Lx, dxm):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    # First orient edges (both paths agree, so use reference)
    edges_o, etri_o = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )

    dxdy_ref, cross_ref = calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o, etri_o, m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, 6400, True
    )
    dxdy_fast, cross_fast = vectorized_calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o, etri_o, m['tcenter'],
        m['xcoord'], m['ycoord']
    )

    assert_array_almost_equal(dxdy_fast, dxdy_ref, decimal=10,
        err_msg=f"edge_dxdy mismatch for Lx={Lx}")
    assert_array_almost_equal(cross_fast, cross_ref, decimal=10,
        err_msg=f"edge_cross_dxdy mismatch for Lx={Lx}")


# =============================================================================
# calculate_laplacian_weights
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_laplacian_weights_consistency(Lx, dxm):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    edges_o, etri_o = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )
    dxdy, cross_dxdy = calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o, etri_o, m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, 6400, True
    )

    pos_ref, num_ref, w_ref, dx_ref = calculate_laplacian_weights(
        m['e2d'], m['ed2d_in'], etri_o, dxdy, cross_dxdy
    )
    pos_fast, num_fast, w_fast, dx_fast = fast_calculate_laplacian_weights(
        m['e2d'], m['ed2d_in'], etri_o, dxdy, cross_dxdy
    )

    assert_array_equal(num_fast, num_ref, err_msg=f"ee_num mismatch for Lx={Lx}")
    assert_array_equal(pos_fast, pos_ref, err_msg=f"ee_pos mismatch for Lx={Lx}")
    assert_array_almost_equal(w_fast, w_ref, decimal=10,
        err_msg=f"weights mismatch for Lx={Lx}")
    assert_array_almost_equal(dx_fast, dx_ref, decimal=10,
        err_msg=f"dxcell mismatch for Lx={Lx}")


# =============================================================================
# build_smoothing_and_metric
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_build_smoothing_consistency(Lx, dxm):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    edges_o, etri_o = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )
    dxdy, cross_dxdy = calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o, etri_o, m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, 6400, True
    )
    ee_pos, ee_num, weights, dxcell = calculate_laplacian_weights(
        m['e2d'], m['ed2d_in'], etri_o, dxdy, cross_dxdy
    )

    sm_ref, met_ref = build_smoothing_and_metric(
        m['e2d'], m['n2d'], ee_num, ee_pos, m['elem_area'], False
    )
    sm_fast, met_fast = fast_build_smoothing_and_metric(
        m['e2d'], m['n2d'], ee_num, ee_pos, m['elem_area'], False
    )

    assert_array_almost_equal(sm_fast, sm_ref, decimal=10,
        err_msg=f"smooth_m mismatch for Lx={Lx}")
    assert_array_almost_equal(met_fast, met_ref, decimal=10,
        err_msg=f"metric mismatch for Lx={Lx}")


# =============================================================================
# assemble_from_intermediate
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_assemble_consistency(Lx, dxm):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    edges_o, etri_o = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )
    dxdy, cross_dxdy = calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o, etri_o, m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, 6400, True
    )
    ee_pos, ee_num, weights, dxcell = calculate_laplacian_weights(
        m['e2d'], m['ed2d_in'], etri_o, dxdy, cross_dxdy
    )
    sm, met = build_smoothing_and_metric(
        m['e2d'], m['n2d'], ee_num, ee_pos, m['elem_area'], False
    )

    ss_ref, ii_ref, jj_ref = assemble_from_intermediate(m['e2d'], ee_num, ee_pos, sm)
    ss_fast, ii_fast, jj_fast = fast_assemble_from_intermediate(m['e2d'], ee_num, ee_pos, sm)

    # Build dense matrices and compare (order of entries may differ)
    from scipy.sparse import coo_matrix
    S_ref = coo_matrix((ss_ref, (ii_ref, jj_ref)), shape=(m['e2d'], m['e2d'])).toarray()
    S_fast = coo_matrix((ss_fast, (ii_fast, jj_fast)), shape=(m['e2d'], m['e2d'])).toarray()

    assert_array_almost_equal(S_fast, S_ref, decimal=10,
        err_msg=f"Assembled sparse matrix mismatch for Lx={Lx}")


# =============================================================================
# End-to-end: full element pipeline
# =============================================================================
@pytest.mark.parametrize("Lx,dxm", MESH_SIZES)
def test_full_element_pipeline(Lx, dxm):
    """
    Run both the original and fast pipelines end-to-end and compare
    the final sparse Laplacian matrix.
    """
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]

    # --- Original pipeline ---
    edges_o1, etri_o1 = orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0
    )
    dxdy1, cross1 = calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o1, etri_o1, m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, 6400, True
    )
    pos1, num1, w1, dc1 = calculate_laplacian_weights(m['e2d'], m['ed2d_in'], etri_o1, dxdy1, cross1)
    sm1, _ = build_smoothing_and_metric(m['e2d'], m['n2d'], num1, pos1, m['elem_area'], False)
    ss1, ii1, jj1 = assemble_from_intermediate(m['e2d'], num1, pos1, sm1)

    # --- Fast pipeline ---
    edges_o2, etri_o2 = vectorized_orient_edges(
        ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord']
    )
    dxdy2, cross2 = vectorized_calculate_dimensional_quantities(
        ed2d, m['ed2d_in'], edges_o2, etri_o2, m['tcenter'],
        m['xcoord'], m['ycoord']
    )
    pos2, num2, w2, dc2 = fast_calculate_laplacian_weights(m['e2d'], m['ed2d_in'], etri_o2, dxdy2, cross2)
    sm2, _ = fast_build_smoothing_and_metric(m['e2d'], m['n2d'], num2, pos2, m['elem_area'], False)
    ss2, ii2, jj2 = fast_assemble_from_intermediate(m['e2d'], num2, pos2, sm2)

    from scipy.sparse import coo_matrix
    S1 = coo_matrix((ss1, (ii1, jj1)), shape=(m['e2d'], m['e2d'])).toarray()
    S2 = coo_matrix((ss2, (ii2, jj2)), shape=(m['e2d'], m['e2d'])).toarray()

    assert_array_almost_equal(S2, S1, decimal=10,
        err_msg=f"Full pipeline Laplacian mismatch for Lx={Lx}")
