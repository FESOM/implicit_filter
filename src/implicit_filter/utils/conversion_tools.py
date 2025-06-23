from implicit_filter import Filter, TriangularFilter, NemoFilter
from ._jax_function import transform_vector_to_nodes, transform_to_nodes
from ._numpy_functions import convert_to_tcells

import jax.numpy as jnp
import numpy as np


def transform_velocity_to_nodes(
    ux: np.ndarray, vy: np.ndarray, filter: Filter
) -> tuple[np.ndarray, np.ndarray]:
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


def transform_to_T_cells(
    ux: np.ndarray, vy: np.ndarray, filter: Filter
) -> tuple[np.ndarray, np.ndarray]:
    if issubclass(filter.__class__, NemoFilter):
        return convert_to_tcells(filter._e2d, filter._ee_pos, ux, uy)
    else:
        raise TypeError("Only NemoFilter and it's subclasses are supported")
