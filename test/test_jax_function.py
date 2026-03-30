import numpy as np
import jax.numpy as jnp
from implicit_filter.utils._jax_function import (
    make_smooth,
    make_smat,
    make_smat_full,
    transform_vector_to_nodes,
    transform_to_nodes,
    transform_to_cells,
    transform_vector_to_cells,
    transform_mask_to_nodes,
    transform_mask_to_elements,
)

def get_basic_mesh():
    n2d = 4
    e2d = 2
    tri = jnp.array([[0, 2, 1], [1, 2, 3]])
    nn_num = jnp.array([2, 3, 3, 2])
    nn_pos = jnp.array([[1, 0, 0, 1],
                        [2, 2, 1, 2],
                        [0, 3, 3, 0]])
    elem_area = jnp.array([0.5, 0.5])
    dx = jnp.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    dy = jnp.array([[0.0, -1.0, 1.0], [0.0, 1.0, -1.0]])
    Mt = jnp.array([1.0, 1.0])
    return n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt

def test_make_smooth():
    n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt = get_basic_mesh()
    smooth_m, metric = make_smooth(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, full=False)
    
    assert smooth_m.shape == nn_pos.shape
    assert metric.shape == nn_pos.shape

def test_make_smat():
    n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt = get_basic_mesh()
    smooth_m, metric = make_smooth(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, full=False)
    nza = int(jnp.sum(nn_num))
    
    ss, ii, jj = make_smat(nn_pos, nn_num, smooth_m, n2d, nza)
    
    assert len(ss) == nza
    assert len(ii) == nza
    assert len(jj) == nza

def test_make_smat_full():
    n2d, e2d, tri, nn_num, nn_pos, elem_area, dx, dy, Mt = get_basic_mesh()
    smooth_m, metric = make_smooth(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, full=True)
    nza = int(jnp.sum(nn_num))
    
    # make_smat_full creates a matrix of size 2*n2d, so nn_pos essentially defines 
    # sparsity graph. The total number of non-zeros for the 2x2 blocks is 4 * nza 
    ss, ii, jj = make_smat_full(nn_pos, nn_num, smooth_m, metric, n2d, nza)
    
    expected_nza = 4 * nza
    assert len(ss) == expected_nza
    assert len(ii) == expected_nza
    assert len(jj) == expected_nza

def test_transform_vector_to_nodes():
    n2d = 4
    e2d = 2
    ux = jnp.array([1.0, 2.0])
    vy = jnp.array([0.5, 1.5])
    ne_pos = jnp.array([[0, 0, 0, 1],
                        [0, 1, 1, 0]])
    ne_num = jnp.array([1, 2, 2, 1])
    elem_area = jnp.array([0.5, 0.5])
    area = jnp.array([0.5/3, 1.0/3, 1.0/3, 0.5/3])
    mask_n = jnp.array([0, 0, 0, 0])

    unod, vnod = transform_vector_to_nodes(ux, vy, ne_pos, ne_num, n2d, elem_area, area, mask_n)
    
    assert unod.shape == (n2d,)
    assert vnod.shape == (n2d,)

def test_transform_to_nodes():
    n2d = 4
    vel = jnp.array([1.0, 2.0])
    ne_pos = jnp.array([[0, 0, 0, 1],
                        [0, 1, 1, 0]])
    ne_num = jnp.array([1, 2, 2, 1])
    elem_area = jnp.array([0.5, 0.5])
    area = jnp.array([0.5/3, 1.0/3, 1.0/3, 0.5/3])
    mask_n = jnp.array([0, 0, 0, 0])

    unod = transform_to_nodes(vel, ne_pos, ne_num, n2d, elem_area, area, mask_n)
    
    assert unod.shape == (n2d,)

def test_transform_to_cells():
    n2d = 4
    e2d = 2
    vel = jnp.array([1.0, 1.0, 2.0, 2.0])
    en_pos = jnp.array([[0, 1],
                        [2, 2],
                        [1, 3]])
    elem_area = jnp.array([0.5, 0.5])

    ucell = transform_to_cells(vel, en_pos, e2d, elem_area)

    assert ucell.shape == (e2d,)

def test_transform_vector_to_cells():
    n2d = 4
    e2d = 2
    ux = jnp.array([1.0, 1.0, 2.0, 2.0])
    vy = jnp.array([1.0, 1.0, 2.0, 2.0])
    en_pos = jnp.array([[0, 1],
                        [2, 2],
                        [1, 3]])
    elem_area = jnp.array([0.5, 0.5])

    ucell, vcell = transform_vector_to_cells(ux, vy, en_pos, e2d, elem_area)

    assert ucell.shape == (e2d,)
    assert vcell.shape == (e2d,)

def test_transform_mask_to_nodes():
    n2d = 4
    mask_ocean = jnp.array([False, True])  # element 0 false (ocean), element 1 true (land)
    ne_pos = jnp.array([[0, 0, 0, 1],
                        [0, 1, 1, 0]])
    ne_num = jnp.array([1, 2, 2, 1])

    mask_n = transform_mask_to_nodes(mask_ocean, ne_pos, ne_num, n2d)
    
    assert mask_n.shape == (n2d,)
    assert not mask_n[0]  # only connects to element 0
    assert mask_n[1]      # connects to element 1 as well
    assert mask_n[2]      # connects to element 1 as well
    assert mask_n[3]      # connects to element 1

def test_transform_mask_to_elements():
    e2d = 2
    mask = jnp.array([True, True, True, False])
    en_pos = jnp.array([[0, 1],
                        [2, 2],
                        [1, 3]])
    
    elem_mask = transform_mask_to_elements(mask, en_pos, e2d)
    
    assert elem_mask.shape == (e2d,)
    # Element 0 connects to nodes 0, 2, 1 -> True
    # Element 1 connects to nodes 1, 2, 3 -> False (node 3 is False)
    assert elem_mask[0]
    assert not elem_mask[1]
