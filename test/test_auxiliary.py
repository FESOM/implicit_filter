import unittest
from implicit_filter.utils._auxiliary import make_tri, neighboring_triangles, neighbouring_nodes
import numpy as np
from numpy.testing import assert_array_equal


class TestMakeTri(unittest.TestCase):
    @staticmethod
    def test_small_case():
        nx = 3
        ny = 3
        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        out = np.array([[0, 1, 3],
                        [1, 4, 3],
                        [1, 2, 4],
                        [2, 5, 4],
                        [3, 4, 6],
                        [4, 7, 6],
                        [4, 5, 7],
                        [5, 8, 7]])

        assert_array_equal(make_tri(nodnum, nx, ny), out)


class TestNeighbouringTriangles(unittest.TestCase):

    def setUp(self) -> None:
        Lx = 5
        Ly = Lx
        dxm = 1
        dym = dxm

        xx = np.arange(0, Lx + 1, dxm)
        yy = np.arange(0, Ly + 1, dym)
        nx = len(xx)
        ny = len(yy)

        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        self.tri = make_tri(nodnum, nx, ny)
        self.n2d = nx * ny
        self.e2d = len(self.tri[:, 1])

    def test_neighboring_triangles(self):
        out_num = np.array([1, 3, 3, 3, 3, 2, 3, 6, 6, 6, 6, 3, 3, 6, 6, 6, 6, 3, 3, 6, 6, 6,
                            6, 3, 3, 6, 6, 6, 6, 3, 2, 3, 3, 3, 3, 1])
        out_pos = np.array([[0, 0, 2, 4, 6, 8, 0, 1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20,
                             21, 23, 25, 27, 29, 30, 31, 33, 35, 37, 39, 40, 41, 43, 45, 47, 49],
                            [0, 1, 3, 5, 7, 9, 1, 2, 4, 6, 8, 18, 11, 12, 14, 16, 18, 28, 21,
                             22, 24, 26, 28, 38, 31, 32, 34, 36, 38, 48, 41, 42, 44, 46, 48, 0],
                            [0, 2, 4, 6, 8, 0, 10, 3, 5, 7, 9, 19, 20, 13, 15, 17, 19, 29, 30,
                             23, 25, 27, 29, 39, 40, 33, 35, 37, 39, 49, 0, 43, 45, 47, 49, 0],
                            [0, 0, 0, 0, 0, 0, 0, 10, 12, 14, 16, 0, 0, 20, 22, 24, 26, 0, 0,
                             30, 32, 34, 36, 0, 0, 40, 42, 44, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 11, 13, 15, 17, 0, 0, 21, 23, 25, 27, 0, 0,
                             31, 33, 35, 37, 0, 0, 41, 43, 45, 47, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 12, 14, 16, 18, 0, 0, 22, 24, 26, 28, 0, 0,
                             32, 34, 36, 38, 0, 0, 42, 44, 46, 48, 0, 0, 0, 0, 0, 0, 0]])

        num, pos = neighboring_triangles(self.n2d, self.e2d, self.tri)
        assert_array_equal(num, out_num)
        assert_array_equal(pos, out_pos)


class TestNeighbouringElements(unittest.TestCase):

    def setUp(self) -> None:
        Lx = 5
        Ly = Lx
        dxm = 1
        dym = dxm

        xx = np.arange(0, Lx + 1, dxm)
        yy = np.arange(0, Ly + 1, dym)
        nx = len(xx)
        ny = len(yy)

        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        self.tri = make_tri(nodnum, nx, ny)
        self.n2d = nx * ny
        self.e2d = len(self.tri[:, 1])
        self.ne_num, self.ne_pos = neighboring_triangles(self.n2d, self.e2d, self.tri)

    def test_neighboring_elements(self):
        out_num = np.array([3, 5, 5, 5, 5, 4, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7,
                            7, 5, 4, 5, 5, 5, 5, 3])
        out_pos = np.array([[0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                            [1, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 11, 7, 13, 14, 15,
                             16, 17, 13, 19, 20, 21, 22, 23, 19, 25, 26, 27, 28, 29, 25, 31, 32, 33, 34, 35],
                            [6, 6, 7, 8, 9, 10, 6, 6, 7, 8, 9, 10, 12, 12, 13, 14,
                             15, 16, 18, 18, 19, 20, 21, 22, 24, 24, 25, 26, 27, 28, 30, 30, 31, 32, 33, 34],
                            [0, 7, 8, 9, 10, 11, 7, 2, 3, 4, 5, 16, 13, 8, 9, 10,
                             11, 22, 19, 14, 15, 16, 17, 28, 25, 20, 21, 22, 23, 34, 31, 26, 27, 28, 29, 0],
                            [0, 2, 3, 4, 5, 0, 12, 8, 9, 10, 11, 17, 18, 14, 15, 16,
                             17, 23, 24, 20, 21, 22, 23, 29, 30, 26, 27, 28, 29, 35, 0, 32, 33, 34, 35, 0],
                            [0, 0, 0, 0, 0, 0, 0, 12, 13, 14, 15, 0, 0, 18, 19, 20,
                             21, 0, 0, 24, 25, 26, 27, 0, 0, 30, 31, 32, 33, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 19, 20, 21,
                             22, 0, 0, 25, 26, 27, 28, 0, 0, 31, 32, 33, 34, 0, 0, 0, 0, 0, 0, 0]])

        num, pos = neighbouring_nodes(self.n2d, self.tri, self.ne_num, self.ne_pos)

        assert_array_equal(num, out_num)
        assert_array_equal(pos, out_pos)


if __name__ == '__main__':
    unittest.main()

import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from implicit_filter.utils._auxiliary import (
    areas,
    convert_to_wavenumbers,
    find_adjacent_points_north,
    find_and_sort_edges_and_triangles,
    calculate_triangle_centers,
    orient_edges,
    create_triangle_to_edge_map,
    calculate_dimensional_quantities,
    calculate_laplacian_weights,
    build_smoothing_and_metric,
    assemble_from_intermediate,
)


def test_convert_to_wavenumbers():
    dist = 100.0
    dxm = 10.0
    expected = 2 * math.pi / (3.5 * (dist / dxm))
    assert convert_to_wavenumbers(dist, dxm) == expected

    with pytest.raises(ValueError):
        convert_to_wavenumbers(0, dxm)

    with pytest.raises(ValueError):
        convert_to_wavenumbers(dist, -5)


def test_areas_carthesian():
    xcoord = np.array([0.0, 1.0, 0.0, 1.0])
    ycoord = np.array([1.0, 1.0, 0.0, 0.0])
    n2d = 4
    e2d = 2
    tri = np.array([[0, 2, 1], [1, 2, 3]])

    ne_num = np.array([1, 2, 2, 1])
    ne_pos = np.array([[0, 0, 0, 1],
                       [0, 1, 1, 0]])

    mask = np.array([1, 1])

    area, elem_area, dx, dy, Mt = areas(
        n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
        meshtype="m", carthesian=True, cyclic_length=0, mask=mask
    )

    assert_array_almost_equal(elem_area, [0.5, 0.5])
    assert_array_almost_equal(area, [0.5 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5 / 3.0])


def test_find_and_sort_edges_and_triangles():
    n2d = 4
    nn_num = np.array([2, 3, 3, 2])
    nn_pos = np.array([[1, 0, 0, 1],
                       [2, 2, 1, 2],
                       [0, 3, 3, 0]])
    ne_num = np.array([1, 2, 2, 1])
    ne_pos = np.array([[0, 0, 0, 1],
                       [0, 1, 1, 0]])

    edges, edge_tri, ed2d_in = find_and_sort_edges_and_triangles(n2d, nn_num, nn_pos, ne_num, ne_pos)
    
    assert ed2d_in == 1  # 1 internal edge (between nodes 1 and 2)
    assert edges.shape[1] == 5  # 5 edges total
    assert edge_tri.shape[1] == 5


def test_calculate_triangle_centers():
    xcoord = np.array([0.0, 1.0, 0.0, 1.0])
    ycoord = np.array([1.0, 1.0, 0.0, 0.0])
    e2d = 2
    tri = np.array([[0, 2, 1], [1, 2, 3]])

    tcenter = calculate_triangle_centers(e2d, tri, xcoord, ycoord, meshtype='m', cyclic_length=0)
    
    assert_array_almost_equal(tcenter[0, 0], (0 + 0 + 1) / 3.0)
    assert_array_almost_equal(tcenter[1, 0], (1 + 0 + 1) / 3.0)


def test_find_adjacent_points_north():
    try:
        import xarray as xr
        import pandas as pd
    except (ImportError, ValueError):
        pytest.skip("Pandas/Xarray not available, skipping test")
        
    x = np.arange(5)
    y = np.arange(5)
            
    lon, lat = np.meshgrid(np.arange(0, 10, 2), np.arange(0, 10, 2))
    lon[-1, :] = lon[-2, :]
    lat[-1, :] = lat[-2, :]

    ds_mm = xr.Dataset(
        {
            "glamt": (["y", "x"], lon),
            "gphit": (["y", "x"], lat),
        },
        coords={
            "x": x,
            "y": y,
        },
    )

    adjacent_x, corresponds = find_adjacent_points_north(ds_mm, 1.0)
    assert corresponds == -2
    assert len(adjacent_x) == 3
    assert_array_equal(adjacent_x.values, [1, 2, 3])


def _north_fold_dataset(nx=40, ny=6, collisions=(2, 4)):
    """A synthetic north fold whose greedy match is imperfect.

    The redundant last row mirrors row -3, so the true correspondence is the
    straight line ``x -> nx - 1 - x``. Each column in ``collisions`` is made
    indistinguishable from its right neighbour at the rounding precision the
    helper matches on, which makes the greedy pass hand that pair out in the
    wrong order -- exactly the rounding damage the linear fit exists to repair.
    The line still has to come back out.
    """
    import numpy as np
    xr = pytest.importorskip("xarray")

    lon = np.tile(np.linspace(0.0, 360.0, nx, endpoint=False), (ny, 1))
    lat = np.tile(np.linspace(-80.0, 80.0, ny)[:, None], (1, nx))
    for j in collisions:
        lon[-3, j] = np.nextafter(lon[-3, j + 1], np.inf)
    lon[-1, :] = lon[-3, ::-1]
    lat[-1, :] = lat[-3, :]
    return xr.Dataset(
        {"glamt": (["y", "x"], lon), "gphit": (["y", "x"], lat)},
        coords={"x": np.arange(nx), "y": np.arange(ny)},
    )


def _north_fold_fit_by_hand(ds, prec):
    """Redo the helper's match, outlier filter and fit in plain numpy.

    Deliberately independent of the implementation: no pandas, and the least
    squares is solved here rather than read back out of the returned Series.
    """
    import numpy as np

    ilon = (np.asarray(ds.glamt) / prec).astype(int)
    redundant, corresponds = ilon[-1, 1:-1], ilon[-3, 1:-1]
    reference = np.arange(1, ilon.shape[1] - 1)

    matched = []                      # the greedy, injective match
    for value in redundant:
        for column, other in zip(reference, corresponds):
            if other == value and column not in matched:
                matched.append(column)
                break
    matched = np.array(matched, dtype=float)

    jumps = np.abs(np.diff(matched, prepend=np.nan))
    keep = jumps < 1.5 * np.nanmean(jumps)          # NaN compares False
    slope, intercept = np.linalg.lstsq(
        np.column_stack([reference[keep], np.ones(keep.sum())]).astype(float),
        matched[keep],
        rcond=None,
    )[0]
    fitted = np.round(slope * reference + intercept).astype(int)
    return reference, matched.astype(int), fitted


def test_north_fold_fit_is_a_least_squares_line():
    """The north-fold helper fits one straight line and rounds it to columns.

    Pins the fit against an independent numpy solve and against the mapping the
    synthetic fold was built with, so that swapping the solver underneath cannot
    change which column each reference column is mapped to.
    """
    import numpy as np
    pytest.importorskip("xarray")
    from implicit_filter.utils._auxiliary import find_adjacent_points_north

    nx, prec = 40, 1e-5
    ds = _north_fold_dataset(nx=nx)
    fit, row = find_adjacent_points_north(ds, prec)

    assert row in (-2, -3)
    values = np.asarray(fit)
    assert values.dtype.kind == "i"

    reference, matched, expected = _north_fold_fit_by_hand(ds, prec)
    assert_array_equal(np.asarray(fit.index), reference)

    # The fold mirrors row -3, so this is the answer independently of any fit.
    assert_array_equal(values, nx - 1 - reference)
    # ... and the rounded least-squares line solved here reproduces it exactly.
    assert_array_equal(values, expected)
    # The greedy match on its own does not: returning it unfitted, bending the
    # fit into a curve or truncating instead of rounding all move the answer.
    assert not np.array_equal(matched, values)


def test_north_fold_fit_refuses_too_few_clean_points():
    """Too few surviving columns must raise, not return a degenerate line.

    ``np.linalg.lstsq`` is happy to solve a rank-deficient system -- an empty
    clean set yields slope 0, intercept 0 and maps every column to 0 -- so the
    helper guards the fit explicitly rather than returning nonsense.
    """
    import numpy as np
    xr = pytest.importorskip("xarray")
    from implicit_filter.utils._auxiliary import find_adjacent_points_north

    nx, ny = 4, 4
    lon = np.tile(np.linspace(0.0, 360.0, nx, endpoint=False), (ny, 1))
    lat = np.tile(np.linspace(-80.0, 80.0, ny)[:, None], (1, nx))
    lon[-1, :] = lon[-3, ::-1]
    lat[-1, :] = lat[-3, :]
    ds = xr.Dataset(
        {"glamt": (["y", "x"], lon), "gphit": (["y", "x"], lat)},
        coords={"x": np.arange(nx), "y": np.arange(ny)},
    )
    with pytest.raises(ValueError, match="too few to fit a line"):
        find_adjacent_points_north(ds, 1e-5)


def test_calculate_laplacian_weights():
    e2d = 2
    ed2d_in = 1
    edge_tri = np.array([
        [0, 0, 0, 1, 1],
        [1, -1, -1, -1, -1]
    ])
    
    edge_dxdy = np.array([[1.0, 0.0, 0.0, 1.0, -1.0],
                          [0.0, 1.0, 0.0, -1.0, 1.0]])
    edge_cross_dxdy = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0]])

    ee_pos, ee_num, weights, dxcell = calculate_laplacian_weights(e2d, ed2d_in, edge_tri, edge_dxdy, edge_cross_dxdy)
    
    assert ee_num[0] == 1
    assert ee_num[1] == 1
    assert weights[0, 0] != 0
    assert weights[0, 1] != 0


def test_build_smoothing_and_metric():
    e2d = 2
    n2d = 4
    ee_num = np.array([1, 1])
    ee_pos = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])
    elem_area = np.array([0.5, 0.5])
    
    smooth_m, metric = build_smoothing_and_metric(e2d, n2d, ee_num, ee_pos, elem_area, full_form=False)
    
    assert smooth_m.shape == (4, 2)
    assert smooth_m[1, 0] < 0  # Off-diagonal
    assert smooth_m[0, 0] > 0  # Diagonal
    assert_array_almost_equal(metric, np.zeros((4, e2d)))


def test_assemble_from_intermediate():
    smooth_m = np.array([
        [2.0, 3.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [0.0, 0.0]
    ])
    ee_num = np.array([2, 1])
    ee_pos = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    e2d = 2

    ss, ii, jj = assemble_from_intermediate(e2d, ee_num, ee_pos, smooth_m)
    
    assert len(ss) == 5
    assert len(ii) == 5
    assert len(jj) == 5
