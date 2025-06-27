from implicit_filter import Filter, TriangularFilter, NemoFilter
from ._jax_function import (
    transform_vector_to_nodes,
    transform_to_nodes,
    transform_mask_to_nodes,
    transform_mask_to_elements,
)
from ._numpy_functions import convert_to_tcells

import jax.numpy as jnp
import numpy as np


def transform_velocity_to_nodes(
    ux: np.ndarray, vy: np.ndarray, filter: Filter
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform velocity components from elements to nodes for triangular meshes.

    Parameters
    ----------
    ux : np.ndarray
        Eastward velocity component defined on mesh elements.
    vy : np.ndarray
        Northward velocity component defined on mesh elements.
    filter : Filter
        Filter instance (must be TriangularFilter or subclass).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Velocity components interpolated to mesh nodes (ux_nodes, vy_nodes).

    Raises
    ------
    TypeError
        If filter is not a TriangularFilter subclass.

    Notes
    -----
    This function performs area-weighted averaging to interpolate velocity
    components from element centers to mesh nodes. Only applicable for
    unstructured triangular meshes.
    """
    if issubclass(filter.__class__, TriangularFilter):
        uxn, vyn = transform_vector_to_nodes(
            jnp.array(ux),
            jnp.array(vy),
            filter._ne_pos,
            filter._ne_num,
            filter._n2d,
            filter._elem_area,
            filter._area,
            filter._mask_n,
        )
        return np.array(uxn), np.array(vyn)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")


def transform_scalar_to_nodes(data, filter: Filter) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform scalar field from elements to nodes for triangular meshes.

    Parameters
    ----------
    data : np.ndarray
        Scalar field defined on mesh elements.
    filter : Filter
        Filter instance (must be TriangularFilter or subclass).

    Returns
    -------
    np.ndarray
        Scalar field interpolated to mesh nodes.

    Raises
    ------
    TypeError
        If filter is not a TriangularFilter subclass.

    Notes
    -----
    Performs area-weighted averaging to interpolate scalar values from
    element centers to mesh nodes. Only applicable for unstructured
    triangular meshes.
    """
    if issubclass(filter.__class__, TriangularFilter):
        uxn = transform_to_nodes(
            jnp.array(data),
            filter._ne_pos,
            filter._ne_num,
            filter._n2d,
            filter._elem_area,
            filter._area,
            filter._mask_n,
        )
        return np.array(uxn)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")


def transform_mask_from_elements_to_nodes(
    mask: np.ndarray, filter: filter
) -> np.ndarray:
    """
    Transform land-sea mask from elements to nodes for triangular meshes.

    Parameters
    ----------
    mask : np.ndarray
        Boolean land-sea mask defined on mesh elements (True = land).
    filter : Filter
        Filter instance (must be TriangularFilter or subclass).

    Returns
    -------
    np.ndarray
        Land-sea mask interpolated to mesh nodes (True = land).

    Raises
    ------
    TypeError
        If filter is not a TriangularFilter subclass.

    Notes
    -----
    A node is considered land if any of its connected elements are land.
    Useful for converting element-based masks to nodal representation.
    """
    if issubclass(filter.__class__, TriangularFilter):
        uxn = transform_mask_to_nodes(
            jnp.array(mask), filter._ne_pos, filter._ne_num, filter._n2d
        )
        return np.array(uxn, dtype=bool)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")


def transform_mask_from_nodes_to_elements(
    mask: np.ndarray, filter: filter
) -> np.ndarray:
    """
    Transform land-sea mask from nodes to elements for triangular meshes.

    Parameters
    ----------
    mask : np.ndarray
        Boolean land-sea mask defined on mesh nodes (True = land).
    filter : Filter
        Filter instance (must be TriangularFilter or subclass).

    Returns
    -------
    np.ndarray
        Land-sea mask interpolated to mesh elements (True = land).

    Raises
    ------
    TypeError
        If filter is not a TriangularFilter subclass.

    Notes
    -----
    An element is considered land if any of its nodes are land.
    Useful for converting nodal masks to element-based representation.
    """
    if issubclass(filter.__class__, TriangularFilter):
        uxn = transform_mask_to_elements(jnp.array(mask), filter._en_pos, filter._e2d)
        return np.array(uxn, dtype=bool)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")


def transform_to_T_cells(
    ux: np.ndarray, vy: np.ndarray, filter: Filter
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform velocity components to T-grid points for NEMO grids.

    Parameters
    ----------
    ux : np.ndarray
        Eastward velocity component on U-grid points.
    vy : np.ndarray
        Northward velocity component on V-grid points.
    filter : Filter
        Filter instance (must be NemoFilter or subclass).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Velocity components interpolated to T-grid points (ux_t, vy_t).

    Raises
    ------
    TypeError
        If filter is not a NemoFilter subclass.

    Notes
    -----
    Specifically designed for NEMO's Arakawa C-grid:
    - U-points: located at east-west cell faces
    - V-points: located at north-south cell faces
    - T-points: located at cell centers
    Performs simple averaging to interpolate velocities to cell centers.
    """
    if issubclass(filter.__class__, NemoFilter):
        return convert_to_tcells(filter._e2d, filter._ee_pos, ux, vy)
    else:
        raise TypeError("Only NemoFilter and it's subclasses are supported")
