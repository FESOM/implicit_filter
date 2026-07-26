Welcome to Implicit Filter's Documentation!
===========================================

.. image:: ../logo.png
   :class: only-light
   :alt: Implicit Filter Logo
   :align: center
   :width: 500px

.. image:: ../logo_dark.png
   :class: only-dark
   :alt: Implicit Filter Logo (Dark Mode)
   :align: center
   :width: 500px

**High-performance spatial filtering for unstructured and structured oceanographic meshes.**

|DOI| |Python| |License|

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg
   :target: https://doi.org/10.5281/zenodo.10907365
.. |Python| image:: https://img.shields.io/badge/python-3.10%2B-blue
.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

Overview
--------

The **Implicit Filter Python Package** provides a suite of classes for filtering data using laplacian-based filters. Allows for low and high-pass filtering on native mesh without loss of resolution. Can efficiently perform spatial spectra extraction. 

Designed for oceanography and climate science, it handles the complexities of various mesh geometries efficiently.

For full mathematical formulation, please refer to our `paper in JAMES <http://dx.doi.org/10.1029/2023MS003946>`_. Implementation details are available at `GMD <https://gmd.copernicus.org/articles/18/6541/2025/>`_.

Key Features
------------

* **Mesh Agnostic**: Can work on triangular or quadrilateral mesh. Native support for **FESOM**, **ICON**, **NEMO**, and regular **Longitude-Latitude** meshes.
* **Element and Node Filtering**: Supports filtering on both mesh nodes and elements (triangles) natively for triangular meshes, automatically adjusting based on input data size.
* **Variable scale filtering**: Filter size can be set individually for each mesh node.
* **GPU Accelerated**: optimized for Nvidia GPUs and Apple Silicon using `JAX <https://jax.readthedocs.io/>`_ for massive performance gains.
* **Efficient**: Optimised for handling even the largest datasets.
* **Smart Caching**: Save and reload computed filter matrices to avoid redundant calculations.

Installation
------------

Standard Installation (CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you do not require GPU acceleration, install directly from GitHub:

.. code-block:: bash

   python -m pip install "implicit_filter[cpu] @ git+https://github.com/FESOM/implicit_filter.git"

GPU Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For optimal performance, usage of an Nvidia GPU or Apple Silicon is highly recommended. The package uses JAX for hardware acceleration.

For CUDA 12.x:

.. code-block:: bash

   python -m pip install "implicit_filter[cuda12] @ git+https://github.com/FESOM/implicit_filter.git"

For CUDA 11.x:

.. code-block:: bash

   python -m pip install "implicit_filter[cuda11] @ git+https://github.com/FESOM/implicit_filter.git"

For Apple Silicon (M1/M2/M3):

.. code-block:: bash

   python -m pip install "implicit_filter[apple] @ git+https://github.com/FESOM/implicit_filter.git"

Quick Start
-----------

Here is a complete example of how to load a FESOM mesh, prepare the filter, and apply it to Scalar data (e.g., SSH).

.. code-block:: python

   from implicit_filter import FesomFilter
   import xarray as xr
   import math

   # 1. Load Data
   path = "/path/to/your/data/"
   mesh_path = path + "fesom.mesh.diag.nc"
   data = xr.open_dataset(path + "ssh.nc")
   unfiltered_data = data['ssh'].values[0, :]

   # 2. Initialize Filter
   flter = FesomFilter()
   flter.prepare_from_file(mesh_path)

   # 2b. Select the backend. Importing implicit_filter pins JAX to CPU, and
   # the `gpu=` argument is not yet implemented, so ask for the GPU here:
   flter.set_backend("gpu")

   # 3. Caching (Optional but Recommended)
   flter.save_to_file("filter_cache")
   # flter = FesomFilter.load_from_file("filter_cache.npz")

   # 4. Define Filter Parameters
   distance = 100  # Target filter size (e.g., km)

   # 5. Apply Filter
   filtered_data = flter.compute(1, 2*math.pi / distance, unfiltered_data)

You can switch between CPU and GPU at runtime using the `set_backend` method:

.. code-block:: python

   flter.set_backend("cpu")
   # or
   flter.set_backend("gpu")

   flter.get_backend()   # -> "cpu" or "gpu"

.. note::

   Importing ``implicit_filter`` sets JAX's platform to CPU for the whole
   process. ``set_backend("gpu")`` is therefore required to use a GPU, and it
   also affects any other JAX code running in the same interpreter.

Filtering on elements
---------------------

For triangular meshes the filter works on **nodes** (vertices) or on
**elements** (triangle centres). The element operator is not built by default;
request it with ``filter_elements=True``:

.. code-block:: python

   flter = FesomFilter()
   flter.prepare_from_file(mesh_path, filter_elements=True)

   filtered_nodes    = flter.compute(1, k, data_on_nodes)
   filtered_elements = flter.compute(1, k, data_on_elements)

Placement is inferred from the length of the data. On a mesh with as many
elements as nodes that is ambiguous, so it can be stated explicitly:

.. code-block:: python

   filtered = flter.compute(1, k, data, on="elements")   # or on="nodes"

The element Laplacian supports two weighting schemes, chosen at ``prepare``
time via ``elem_weights``:

``'equilateral'`` (default)
   Fixed ``sqrt(3)/area`` coefficient — the finite-volume weight for an
   equilateral triangle. Depends only on cell area and ignores cell shape.
   This is the long-standing behaviour, kept as the default so that existing
   results remain reproducible.

``'geometric'``
   Uses the per-edge weights computed from the actual mesh geometry. Identical
   to ``'equilateral'`` on an equilateral mesh; more accurate on anisotropic or
   strongly graded meshes.

For advanced performance, you can warm-start the iterative solver if you have a good initial guess (e.g., from a previous time step):

.. code-block:: python

   filtered_data = flter.compute(1, 2*math.pi / distance, unfiltered_data, x0=previous_guess)

Examples Gallery
----------------

The ``implicit_filter`` package provides several interactive Jupyter notebook examples demonstrating its capabilities across different grids and use cases.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 📐 Basic Triangular Mesh (Nodes)
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/basic_triangular_mesh.ipynb

      A quick start tutorial on how to apply the filter to spatial data defined on unstructured triangular meshes.

   .. grid-item-card:: 📐 Basic Triangular Mesh (Elements)
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/element_filtering_example.ipynb

      A quick start tutorial on how to apply the filter to spatial data defined on the elements level of unstructured triangular meshes.

   .. grid-item-card:: 🌍 Latitude-Longitude Grid
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/lat_long_grid_example.ipynb

      Walkthrough of applying the filter on structured standard Latitude-Longitude grids typical for many ocean models.

   .. grid-item-card:: 🎛️ Variable Filter Scales
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/variable_filter_scales.ipynb

      Shows how to use the package's unique capability to apply a filter where the smoothing scale varies spatially depending on the region.

   .. grid-item-card:: 🎭 Data Masking
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/data_masking_example.ipynb

      Illustrates techniques for properly masking out land or missing data values before and after filtering.

   .. grid-item-card:: 🐟 NEMO Example
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/nemo_example.ipynb

      An example demonstrating the usage of the filter specifically tailored for NEMO ocean model outputs.

   .. grid-item-card:: 📊 Spectra Extraction
      :link: https://github.com/FESOM/implicit_filter/blob/main/examples/spectra_extraction.ipynb

      Demonstrates how to effectively extract spatial spectra from the filtered model data.

Support & Feature Requests
--------------------------

**Missing a feature? Using a model grid that isn't supported yet?**

I am actively developing this package and am always happy to help! If you are interested in using Implicit Filter but find something missing:

1. **Open an Issue**: Please describe your use case or the mesh you are using.
2. **Get Quick Support**: Adding support for new meshes or implementing specific features can often be done quickly.

Don't hesitate to reach out—feedback and new use cases are highly appreciated! 🚀

Developer Setup
---------------

Requires Python 3.10+.

.. code-block:: bash

   git clone https://github.com/FESOM/implicit_filter
   cd implicit_filter
   pip install -e .

Citation
--------

If you use this package in your research, please cite: https://doi.org/10.5194/gmd-18-6541-2025

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Developer Reference

   api
