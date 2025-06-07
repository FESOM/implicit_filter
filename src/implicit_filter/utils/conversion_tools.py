from implicit_filter import Filter, TriangularFilter
from ._jax_function import transform_vector_to_nodes, transform_to_nodes

import jax.numpy as jnp
import numpy as np


def transform_velocity_to_nodes(ux, vy, filter: Filter) -> tuple[np.ndarray, np.ndarray]:
    if issubclass(filter.__class__, TriangularFilter):
        uxn, vyn = transform_vector_to_nodes(jnp.array(ux), jnp.array(vy), filter._en_pos, filter._ne_num, filter._n2d,
                                             filter._elem_area, filter._area)
        return np.array(uxn), np.array(vyn)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")


def transform_scalar_to_nodes(data, filter: Filter) -> tuple[np.ndarray, np.ndarray]:
    if issubclass(filter.__class__, TriangularFilter):
        uxn, vyn = transform_to_nodes(jnp.array(data), filter._en_pos, filter._ne_num, filter._n2d,
                                      filter._elem_area, filter._area)
        return np.array(uxn), np.array(vyn)
    else:
        raise TypeError("Only TriangularFilter and it's subclasses are supported")
