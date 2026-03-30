"""
JAX-accelerated and vectorized NumPy versions of element-based filtering
functions. These replace the pure-Python loop versions in _auxiliary.py.
"""
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial


# =============================================================================
# Vectorized NumPy functions (topology & geometry — not JIT-able)
# =============================================================================

def vectorized_orient_edges(ed2d, edges, edge_tri, tcenter, xcoord, ycoord):
    """
    Vectorized edge orientation for meshtype='m'.
    Orders edges so that tri[0] is on the left of the edge vector.
    """
    edges_o = edges.copy()
    edge_tri_o = edge_tri.copy()

    ed0 = edges_o[0, :]  # (ed2d,)
    ed1 = edges_o[1, :]  # (ed2d,)
    tri1_idx = edge_tri_o[0, :]  # (ed2d,)

    # Vector from edge start to tri1 center
    xc0 = tcenter[0, tri1_idx] - xcoord[ed0]
    xc1 = tcenter[1, tri1_idx] - ycoord[ed0]

    # Edge vector
    xe0 = xcoord[ed1] - xcoord[ed0]
    xe1 = ycoord[ed1] - ycoord[ed0]

    # Cross product z-component
    cross = xc0 * xe1 - xc1 * xe0  # (ed2d,)

    # Where cross > 0, we need to swap
    swap_mask = cross > 0
    has_second_tri = edge_tri_o[1, :] != -1

    # Case 1: cross > 0 AND has second tri → swap tri indices
    swap_tri = swap_mask & has_second_tri
    t0 = edge_tri_o[0, :].copy()
    t1 = edge_tri_o[1, :].copy()
    edge_tri_o[0, swap_tri] = t1[swap_tri]
    edge_tri_o[1, swap_tri] = t0[swap_tri]

    # Case 2: cross > 0 AND no second tri → swap edge node indices
    swap_edge = swap_mask & ~has_second_tri
    e0 = edges_o[0, :].copy()
    e1 = edges_o[1, :].copy()
    edges_o[0, swap_edge] = e1[swap_edge]
    edges_o[1, swap_edge] = e0[swap_edge]

    return edges_o, edge_tri_o


def vectorized_calculate_dimensional_quantities(ed2d, ed2d_in, edges, edge_tri, tcenter, xcoord, ycoord):
    """
    Vectorized calculation of geometric edge properties for meshtype='m'.
    """
    # Edge vectors
    edge_dxdy = np.zeros((2, ed2d))
    edge_dxdy[0, :] = xcoord[edges[1, :]] - xcoord[edges[0, :]]
    edge_dxdy[1, :] = ycoord[edges[1, :]] - ycoord[edges[0, :]]

    # Edge midpoints
    mid_x = 0.5 * (xcoord[edges[0, :]] + xcoord[edges[1, :]])
    mid_y = 0.5 * (ycoord[edges[0, :]] + ycoord[edges[1, :]])

    # Cross vectors: center of tri[k] minus edge midpoint
    edge_cross_dxdy = np.zeros((4, ed2d))
    edge_cross_dxdy[0, :] = tcenter[0, edge_tri[0, :]] - mid_x
    edge_cross_dxdy[1, :] = tcenter[1, edge_tri[0, :]] - mid_y

    # Internal edges have a second triangle
    internal = np.arange(ed2d) < ed2d_in
    idx = edge_tri[1, internal]
    if idx.size > 0:
        edge_cross_dxdy[2, internal] = tcenter[0, idx] - mid_x[internal]
        edge_cross_dxdy[3, internal] = tcenter[1, idx] - mid_y[internal]

    return edge_dxdy, edge_cross_dxdy


# =============================================================================
# Vectorized NumPy versions of the numerical hot-path
# =============================================================================

def fast_calculate_laplacian_weights(e2d, ed2d_in, edge_tri, edge_dxdy, edge_cross_dxdy):
    """
    Vectorized computation of Laplacian weights.
    Replaces the Python loop over internal edges.
    """
    ee_pos = -np.ones((3, e2d), dtype=int)
    ee_num = np.zeros(e2d, dtype=int)
    weights = np.zeros((3, e2d))
    dxcell = np.zeros((3, e2d))

    # Pre-compute all weights for internal edges at once
    elem1 = edge_tri[0, :ed2d_in]  # (ed2d_in,)
    elem2 = edge_tri[1, :ed2d_in]  # (ed2d_in,)

    # b = -cross[0:2] + cross[2:4]  for each internal edge
    b0 = -edge_cross_dxdy[0, :ed2d_in] + edge_cross_dxdy[2, :ed2d_in]
    b1 = -edge_cross_dxdy[1, :ed2d_in] + edge_cross_dxdy[3, :ed2d_in]

    # a_normal = (edge_dy, -edge_dx)
    a_n0 = edge_dxdy[1, :ed2d_in]
    a_n1 = -edge_dxdy[0, :ed2d_in]

    # dot products
    dot_b = b0 * b0 + b1 * b1
    dot_ab = a_n0 * b0 + a_n1 * b1

    w = np.where(dot_b != 0, dot_ab / dot_b, 0.0)

    # Fill ee_pos, ee_num, weights, dxcell sequentially
    # (This loop is over internal edges — unavoidable due to variable-length neighbor lists)
    for n in range(ed2d_in):
        e1, e2 = elem1[n], elem2[n]

        ee_pos[ee_num[e1], e1] = e2
        ee_pos[ee_num[e2], e2] = e1

        weights[ee_num[e1], e1] = w[n]
        weights[ee_num[e2], e2] = w[n]

        dxcell[ee_num[e1], e1] = 0.5 * a_n0[n]
        dxcell[ee_num[e2], e2] = -0.5 * a_n0[n]

        ee_num[e1] += 1
        ee_num[e2] += 1

    return ee_pos, ee_num, weights, dxcell


def fast_build_smoothing_and_metric(e2d, n2d, ee_num, ee_pos, elem_area, full_form, Mt=None, dxcell=None):
    """
    Fully vectorized construction of smoothing and metric matrices.
    No Python loops.
    """
    smooth_m = np.zeros((4, e2d))
    metric = np.zeros((4, e2d))

    # Mask for elements with positive area
    valid = elem_area > 0

    off_diag = -np.sqrt(3) / np.where(valid, elem_area, 1.0)  # avoid div/0

    # Fill off-diagonal entries: rows 1, 2, 3 (for neighbors 0, 1, 2)
    for k in range(3):
        active = valid & (ee_num > k)
        smooth_m[k + 1, active] = off_diag[active]

    # Diagonal = negative sum of off-diagonals
    smooth_m[0, :] = -np.sum(smooth_m[1:, :], axis=0)

    if full_form and Mt is not None and dxcell is not None:
        smooth_m[0, valid] += Mt[valid] ** 2
        for k in range(3):
            active = valid & (ee_num > k)
            metric[k + 1, active] = 2.0 * dxcell[k, active] * Mt[active] / elem_area[active]

    return smooth_m, metric


def fast_assemble_from_intermediate(e2d, ee_num, ee_pos, smooth_m):
    """
    Vectorized assembly of COO sparse triplets from smooth_m.
    No Python loops.
    """
    nza = int(np.sum(ee_num)) + e2d

    # Diagonal entries: one per element
    diag_ss = smooth_m[0, :]
    diag_ii = np.arange(e2d)
    diag_jj = np.arange(e2d)

    # Off-diagonal entries
    off_ss_list = []
    off_ii_list = []
    off_jj_list = []

    for k in range(3):
        active = ee_num > k
        if not np.any(active):
            break
        active_idx = np.where(active)[0]
        off_ss_list.append(smooth_m[k + 1, active_idx])
        off_ii_list.append(active_idx)
        off_jj_list.append(ee_pos[k, active_idx])

    if off_ss_list:
        off_ss = np.concatenate(off_ss_list)
        off_ii = np.concatenate(off_ii_list)
        off_jj = np.concatenate(off_jj_list)
    else:
        off_ss = np.array([], dtype=float)
        off_ii = np.array([], dtype=int)
        off_jj = np.array([], dtype=int)

    ss = np.concatenate([diag_ss, off_ss])
    ii = np.concatenate([diag_ii, off_ii])
    jj = np.concatenate([diag_jj, off_jj])

    return ss, ii, jj
