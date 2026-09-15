=============
API Reference
=============

This page provides detailed documentation for all modules and classes in the ``implicit_filter`` package.

Filter Base Class
-----------------

.. automodule:: implicit_filter.filter
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Triangular Mesh Filters
-----------------------

.. automodule:: implicit_filter.triangular_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: implicit_filter.fesom_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: implicit_filter.icon_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: implicit_filter.reduced_gaussian_filter
   :members:
   :undoc-members:
   :show-inheritance:

Structured Grid Filters
-----------------------

.. automodule:: implicit_filter.latlon_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: implicit_filter.nemo_filter
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: implicit_filter
   :members: transform_velocity_to_nodes, 
             transform_scalar_to_nodes,
             transform_mask_from_elements_to_nodes,
             transform_mask_from_nodes_to_elements,
             transform_to_T_cells,
             convert_to_wavenumbers,
             gaussian_latitudes,
             reduced_gaussian_grid,
             spherical_triangulation
   :noindex:

.. autofunction:: implicit_filter.transform_velocity_to_nodes

.. autofunction:: implicit_filter.transform_scalar_to_nodes

.. autofunction:: implicit_filter.transform_mask_from_elements_to_nodes

.. autofunction:: implicit_filter.transform_mask_from_nodes_to_elements

.. autofunction:: implicit_filter.transform_to_T_cells

.. autofunction:: implicit_filter.convert_to_wavenumbers

.. autofunction:: implicit_filter.gaussian_latitudes

.. autofunction:: implicit_filter.reduced_gaussian_grid

.. autofunction:: implicit_filter.spherical_triangulation

Submodules
----------

* :mod:`implicit_filter.utils`

.. automodule:: implicit_filter.utils
   :members:
   :undoc-members:
   :show-inheritance:
