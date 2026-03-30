#!/usr/bin/env python
"""
Benchmark: compare original (pure-Python loop) vs fast (vectorized)
element filtering functions across multiple mesh sizes.
"""
import time
import numpy as np
from implicit_filter.utils._auxiliary import (
    make_tri, neighboring_triangles, neighbouring_nodes, areas,
    find_and_sort_edges_and_triangles, calculate_triangle_centers,
    orient_edges, calculate_dimensional_quantities,
    calculate_laplacian_weights, build_smoothing_and_metric,
    assemble_from_intermediate,
)
from implicit_filter.utils._jax_elem_function import (
    vectorized_orient_edges,
    vectorized_calculate_dimensional_quantities,
    fast_calculate_laplacian_weights,
    fast_build_smoothing_and_metric,
    fast_assemble_from_intermediate,
)


def build_mesh(Lx, dxm=1):
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
        "m", True, 0, mask,
    )
    edges, edge_tri, ed2d_in = find_and_sort_edges_and_triangles(
        n2d, nn_num, nn_pos, ne_num, ne_pos
    )
    tcenter = calculate_triangle_centers(e2d, tri, xcoord, ycoord, 'm', 0)
    return {
        'n2d': n2d, 'e2d': e2d, 'tri': tri,
        'xcoord': xcoord, 'ycoord': ycoord,
        'nn_num': nn_num, 'nn_pos': nn_pos,
        'ne_num': ne_num, 'ne_pos': ne_pos,
        'elem_area': elem_area, 'Mt': Mt,
        'edges': edges, 'edge_tri': edge_tri, 'ed2d_in': ed2d_in,
        'tcenter': tcenter, 'dx': dx, 'dy': dy,
    }


def time_fn(fn, *args, repeats=5):
    """Time a function, return median time in ms."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), result


def benchmark_mesh(Lx, dxm=1):
    m = build_mesh(Lx, dxm)
    ed2d = m['edges'].shape[1]
    repeats = 10

    print(f"\n{'='*70}")
    print(f"  Mesh: Lx={Lx}, dxm={dxm}  →  n2d={m['n2d']}, e2d={m['e2d']}, edges={ed2d}")
    print(f"{'='*70}")
    print(f"{'Function':<40} {'Original':>10} {'Fast':>10} {'Speedup':>10}")
    print(f"{'-'*70}")

    # --- orient_edges ---
    t_orig, (eo1, et1) = time_fn(
        orient_edges, ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], 'm', 0, repeats=repeats
    )
    t_fast, _ = time_fn(
        vectorized_orient_edges, ed2d, m['edges'], m['edge_tri'], m['tcenter'],
        m['xcoord'], m['ycoord'], repeats=repeats
    )
    print(f"{'orient_edges':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")

    # --- calculate_dimensional_quantities ---
    t_orig, (dxdy1, cross1) = time_fn(
        calculate_dimensional_quantities, ed2d, m['ed2d_in'], eo1, et1,
        m['tcenter'], m['xcoord'], m['ycoord'], 'm', 0, 6400, True,
        repeats=repeats
    )
    t_fast, _ = time_fn(
        vectorized_calculate_dimensional_quantities, ed2d, m['ed2d_in'], eo1, et1,
        m['tcenter'], m['xcoord'], m['ycoord'], repeats=repeats
    )
    print(f"{'calculate_dimensional_quantities':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")

    # --- calculate_laplacian_weights ---
    t_orig, (pos1, num1, w1, dc1) = time_fn(
        calculate_laplacian_weights, m['e2d'], m['ed2d_in'], et1, dxdy1, cross1,
        repeats=repeats
    )
    t_fast, _ = time_fn(
        fast_calculate_laplacian_weights, m['e2d'], m['ed2d_in'], et1, dxdy1, cross1,
        repeats=repeats
    )
    print(f"{'calculate_laplacian_weights':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")

    # --- build_smoothing_and_metric ---
    t_orig, (sm1, _) = time_fn(
        build_smoothing_and_metric, m['e2d'], m['n2d'], num1, pos1, m['elem_area'], False,
        repeats=repeats
    )
    t_fast, _ = time_fn(
        fast_build_smoothing_and_metric, m['e2d'], m['n2d'], num1, pos1, m['elem_area'], False,
        repeats=repeats
    )
    print(f"{'build_smoothing_and_metric':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")

    # --- assemble_from_intermediate ---
    t_orig, _ = time_fn(
        assemble_from_intermediate, m['e2d'], num1, pos1, sm1,
        repeats=repeats
    )
    t_fast, _ = time_fn(
        fast_assemble_from_intermediate, m['e2d'], num1, pos1, sm1,
        repeats=repeats
    )
    print(f"{'assemble_from_intermediate':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")

    # --- Full pipeline ---
    def original_pipeline():
        eo, et = orient_edges(ed2d, m['edges'], m['edge_tri'], m['tcenter'],
                              m['xcoord'], m['ycoord'], 'm', 0)
        dd, cc = calculate_dimensional_quantities(ed2d, m['ed2d_in'], eo, et,
                                                  m['tcenter'], m['xcoord'], m['ycoord'], 'm', 0, 6400, True)
        p, n, w, d = calculate_laplacian_weights(m['e2d'], m['ed2d_in'], et, dd, cc)
        s, _ = build_smoothing_and_metric(m['e2d'], m['n2d'], n, p, m['elem_area'], False)
        return assemble_from_intermediate(m['e2d'], n, p, s)

    def fast_pipeline():
        eo, et = vectorized_orient_edges(ed2d, m['edges'], m['edge_tri'], m['tcenter'],
                                         m['xcoord'], m['ycoord'])
        dd, cc = vectorized_calculate_dimensional_quantities(ed2d, m['ed2d_in'], eo, et,
                                                             m['tcenter'], m['xcoord'], m['ycoord'])
        p, n, w, d = fast_calculate_laplacian_weights(m['e2d'], m['ed2d_in'], et, dd, cc)
        s, _ = fast_build_smoothing_and_metric(m['e2d'], m['n2d'], n, p, m['elem_area'], False)
        return fast_assemble_from_intermediate(m['e2d'], n, p, s)

    t_orig, _ = time_fn(original_pipeline, repeats=repeats)
    t_fast, _ = time_fn(fast_pipeline, repeats=repeats)
    print(f"{'-'*70}")
    print(f"{'TOTAL PIPELINE':<40} {t_orig:>8.2f}ms {t_fast:>8.2f}ms {t_orig/t_fast:>8.1f}x")


if __name__ == '__main__':
    for Lx in [20, 50, 100, 200]:
        benchmark_mesh(Lx)
