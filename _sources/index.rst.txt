.. implicit_filter documentation master file

Welcome to Implicit Filter's Documentation!
===========================================

.. image:: logo.png
   :alt: Implicit Filter Logo
   :align: center
   :width: 300px

|

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg
   :target: https://doi.org/10.5281/zenodo.10907365
   :alt: DOI

.. image:: https://img.shields.io/badge/arXiv-2404.07398-b31b1b.svg
   :target: https://arxiv.org/abs/2404.07398
   :alt: arXiv

The Implicit Filter Python Package provides efficient filtering implementations for ocean model data using
implicit filtering techniques. Supports FESOM, ICON, NEMO, and longitude-latitude grids with GPU acceleration.

**Key Features**:

- Optimized for Nvidia GPU acceleration (highly recommended)
- Support for major ocean model grids (FESOM, ICON, NEMO, lat-lon)
- Filter state caching for improved performance
- Spectral analysis capabilities

Mathematical Background
-----------------------
For implementation details, see our paper on `arXiv <https://arxiv.org/abs/2404.07398>`_.

For the full mathematical formulation, refer to:
`Original Methodology Paper <http://dx.doi.org/10.1029/2023MS003946>`_

Installation
------------

End-user Installation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create environment with cupy (e.g. from Conda)
   # Activate env
   python -m pip install git+https://github.com/FESOM/implicit_filter.git

For GPU acceleration, install CuPy matching your CUDA version:

.. code-block:: bash

   # CUDA 11
   pip install cupy-cuda11x

   # CUDA 12 or newer
   pip install cupy-cuda12x

   # Or install with package (CUDA 12 example):
   python -m pip install "implicit_filter[gpu_c12] @ git+https://github.com/FESOM/implicit_filter.git"

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/FESOM/implicit_filter
   cd implicit_filter
   pip install -e .  # CPU-only installation

Dependencies
~~~~~~~~~~~~

+----------------+--------------------------------+
| CPU Required   | NumPy, SciPy, JAX             |
+----------------+--------------------------------+
| GPU Required   | CuPy, JAX                     |
+----------------+--------------------------------+
| Visualization  | Matplotlib                    |
+----------------+--------------------------------+
| I/O            | Xarray                        |
+----------------+--------------------------------+

Quick Start Tutorial
-------------------

Basic scalar filtering example:

.. code-block:: python

   import xarray as xr
   from implicit_filter import FesomFilter, convert_to_wavenumbers

   # Load data
   path = "your/path/"
   mesh = xr.open_dataset(path + "fesom.mesh.diag.nc")
   data = xr.open_dataset(path + "ssh.nc")
   unfiltered = data['ssh'].values[0, :]

   # Create filter (use GPU if available)
   flter = FesomFilter()
   flter.prepare_from_file(path + "fesom.mesh.diag.nc", gpu=True)

   # Save filter state for reuse
   flter.save_to_file("filter_cache.npz")

   # Calculate wavenumber (100km filter, 5km resolution)
   k = convert_to_wavenumbers(100, 5)

   # Apply filter
   filtered = flter.compute(1, k, unfiltered)

For more advanced examples including velocity filtering and spectral analysis, see the :doc:`examples`.

API Reference
-------------

Core Classes:

.. toctree::
   :maxdepth: 2

   filter
   triangular_filter
   fesom_filter
   icon_filter
   latlon_filter
   nemo_filter

Utility Functions:

.. toctree::
   :maxdepth: 2

   utils

Examples
--------

.. toctree::
   :maxdepth: 2

   examples

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
