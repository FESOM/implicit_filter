import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from implicit_filter.utils._numpy_functions import (
    make_smooth,
    make_smat,
    convert_to_tcells,
    calculate_global_nemo_neighbourhood,
    calculate_global_regular_neighbourhood,
    calculate_local_regular_neighbourhood,
)

def get_basic_mesh():
    n2d = 4
    e2d = 2
    tri = np.array([[0, 2, 1], [1, 2, 3]])
    nn_num = np.array([2, 3, 3, 2])
    nn_pos = np.array([[1, 0, 0, 1],
                       [2, 2, 1, 2],
                       [0, 3, 3, 0]])
    elem_area = np.array([0.5, 0.5])
    dx = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    dy = np.array([[0.0, -1.0, 1.0], [0.0, 1.0, -1.0]])
    Mt = np.array([1.0, 1.0])
    return n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt

def test_make_smooth():
    n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt = get_basic_mesh()
    
    smooth_m = make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)
    
    assert smooth_m.shape == nn_pos.shape
    # Check that diagonal is non-zero
    assert smooth_m[0, 0] != 0.0

def test_make_smat():
    n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt = get_basic_mesh()
    smooth_m = make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)
    nza = int(np.sum(nn_num))
    
    ss, ii, jj = make_smat(nn_pos, nn_num, smooth_m, n2d, nza)
    
    assert len(ss) == nza

def test_convert_to_tcells():
    e2d = 2 
    ux = np.array([1.0, 2.0])
    vy = np.array([0.5, 1.5])
    # ee_pos contains neighbors. 4 neighbors per element.
    ee_pos = np.array([[0, 1],
                       [1, 0],
                       [0, 1],
                       [1, 0]])
    
    tu, tv = convert_to_tcells(e2d, ee_pos, ux, vy)
    
    assert tu.shape == ux.shape
    assert tv.shape == vy.shape
    
    # According to logic: v[0, i] = u[i], v[2, i] = u[pos[2, i]] 
    # The output is basically an average across faces. It should be numerically stable.
    assert not np.isnan(tu).any()

def test_calculate_global_nemo_neighbourhood():
    nx, ny = 3, 3
    e2d = nx * ny
    north_adj = np.array([0, 2, 1, 0]) # Mock north map (1-indexed theoretically for nemo but we mock)
    
    ee_pos, nza = calculate_global_nemo_neighbourhood(e2d, nx, ny, north_adj)
    
    assert ee_pos.shape == (4, e2d)
    assert nza > 0

def test_calculate_global_regular_neighbourhood():
    nx, ny = 3, 3
    e2d = nx * ny
    
    ee_pos, nza = calculate_global_regular_neighbourhood(e2d, nx, ny)
    
    assert ee_pos.shape == (4, e2d)
    assert nza > 0
    # Left border of x=0 wrapper expects to connect to x=nx-1
    assert ee_pos[0, 0] == ny * (nx - 1) 

def test_calculate_local_regular_neighbourhood():
    nx, ny = 3, 3
    e2d = nx * ny
    
    ee_pos, nza = calculate_local_regular_neighbourhood(e2d, nx, ny)
    
    assert ee_pos.shape == (4, e2d)
    assert nza > 0
    # Corner does not wrap around
    assert ee_pos[0, 0] == 0 
