import numpy as np
import pytest
from implicit_filter import TriangularFilter

def test_element_filtering_init():
    # Make a simple mesh
    n2d = 4
    e2d = 2
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    xcoord = np.array([0.0, 1.0, 0.0, 1.0])
    ycoord = np.array([0.0, 0.0, 1.0, 1.0])
    
    filt = TriangularFilter()
    # It should work without filter_elements
    filt.prepare(n2d, e2d, tri, xcoord, ycoord, meshtype='m', cartesian=True, full=False, filter_elements=False)
    
    # Try filtering elements, should raise ValueError
    elem_data = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        filt.compute(1, 1.0, elem_data)
        
    # Re-prepare with filter_elements=True
    filt.prepare(n2d, e2d, tri, xcoord, ycoord, meshtype='m', cartesian=True, full=False, filter_elements=True)
    
    # Now it should work
    res = filt.compute(1, 1.0, elem_data)
    assert len(res) == e2d
    
    # And node filtering should still work
    node_data = np.array([1.0, 2.0, 3.0, 4.0])
    res_node = filt.compute(1, 1.0, node_data)
    assert len(res_node) == n2d

def test_element_filtering_spectra():
    n2d = 4
    e2d = 2
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    xcoord = np.array([0.0, 1.0, 0.0, 1.0])
    ycoord = np.array([0.0, 0.0, 1.0, 1.0])
    
    filt = TriangularFilter()
    filt.prepare(n2d, e2d, tri, xcoord, ycoord, meshtype='m', cartesian=True, full=False, filter_elements=True)
    
    elem_data = np.array([1.0, 2.0])
    k = np.array([1.0])
    
    spectra = filt.compute_spectra_scalar(1, k, elem_data)
    assert len(spectra) == 2

    # Vector
    ux = np.array([1.0, 2.0])
    vy = np.array([0.0, -1.0])
    spectra_v = filt.compute_spectra_velocity(1, k, ux, vy)
    assert len(spectra_v) == 2
