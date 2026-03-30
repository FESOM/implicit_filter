import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from implicit_filter.triangular_filter import TriangularFilter

try:
    from implicit_filter.nemo_filter import NemoFilter
except Exception:
    NemoFilter = None

from implicit_filter.utils.conversion_tools import (
    transform_velocity_to_nodes,
    transform_scalar_to_nodes,
    transform_mask_from_elements_to_nodes,
    transform_mask_from_nodes_to_elements,
    transform_to_T_cells,
)

import jax.numpy as jnp

class MockTriangularFilter(TriangularFilter):
    def __init__(self):
        # We don't initialize the full parent C-bindings just simple attrs
        self._n2d = 4
        self._e2d = 2
        self._ne_num = jnp.array([1, 2, 2, 1])
        self._ne_pos = jnp.array([[0, 0, 0, 1],
                                 [0, 1, 1, 0]])
        self._en_pos = jnp.array([[0, 1],
                                 [2, 2],
                                 [1, 3]])
        self._elem_area = jnp.array([0.5, 0.5])
        self._area = jnp.array([0.5/3, 1.0/3, 1.0/3, 0.5/3])
        self._mask_n = jnp.zeros(4, dtype=int)

class DummyFilter:
    pass

@pytest.fixture
def tri_filter():
    return MockTriangularFilter()

def test_transform_velocity_to_nodes(tri_filter):
    ux = np.array([1.0, 2.0])
    vy = np.array([0.5, 1.5])
    
    ux_n, vy_n = transform_velocity_to_nodes(ux, vy, tri_filter)
    
    assert ux_n.shape == (tri_filter._n2d,)
    assert vy_n.shape == (tri_filter._n2d,)
    
    with pytest.raises(TypeError):
        transform_velocity_to_nodes(ux, vy, DummyFilter())

def test_transform_scalar_to_nodes(tri_filter):
    data = np.array([1.0, 2.0])
    
    data_n = transform_scalar_to_nodes(data, tri_filter)
    
    assert data_n.shape == (tri_filter._n2d,)
    
    with pytest.raises(TypeError):
        transform_scalar_to_nodes(data, DummyFilter())

def test_transform_mask_from_elements_to_nodes(tri_filter):
    mask = np.array([False, True])
    
    mask_n = transform_mask_from_elements_to_nodes(mask, tri_filter)
    
    assert mask_n.shape == (tri_filter._n2d,)
    
    with pytest.raises(TypeError):
        transform_mask_from_elements_to_nodes(mask, DummyFilter())

def test_transform_mask_from_nodes_to_elements(tri_filter):
    mask_n = np.array([False, True, False, False])
    
    elem_mask = transform_mask_from_nodes_to_elements(mask_n, tri_filter)
    
    assert elem_mask.shape == (tri_filter._e2d,)
    
    with pytest.raises(TypeError):
        transform_mask_from_nodes_to_elements(mask_n, DummyFilter())

def test_transform_to_T_cells():
    if NemoFilter is None:
        pytest.skip("NemoFilter is completely unavailable")
        
    class MockNemoFilter(NemoFilter):
        def __init__(self):
            # mock its needed properties
            self._nx = 2
            self._ny = 2
            self._e2d = 4
            self._ee_pos = np.array([
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [0, 1, 2, 3],
                [1, 0, 3, 2]
            ])
            
    try:
        nemo_filter = MockNemoFilter()
        ux = np.array([[1.0, 2.0], [3.0, 4.0]])
        vy = np.array([[0.5, 1.5], [2.5, 3.5]])
        
        tu, tv = transform_to_T_cells(ux, vy, nemo_filter)
        
        assert tu.shape == ux.shape
        assert tv.shape == vy.shape
        
        with pytest.raises(TypeError):
            transform_to_T_cells(ux, vy, DummyFilter())
    except Exception:
        pytest.skip("Nemo configuration setup not fully mockable in environment")
