<div align="center">

<img src="logo.png" alt="Implicit Filter Logo" width="500"/>



**High-performance spatial filtering for unstructured and structured oceanographic meshes.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg)](https://doi.org/10.5281/zenodo.10907365)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)



</div>

---

## 🌊 Overview

The **Implicit Filter Python Package** provides a suite of classes for filtering data using laplacian-based filters. Allows for low and high-pass filtering on native mesh without loss of resolution. Can efficiently perform spatial spectra extraction 

 Designed for oceanography and climate science, it handles the complexities of various mesh geometries efficiently.

For full mathematical formulation, please refer to our [paper in JAMES](http://dx.doi.org/10.1029/2023MS003946). Implementation details are available at [GMD](https://gmd.copernicus.org/articles/18/6541/2025/).

## ✨ Key Features
*   **🌐 Mesh Agnostic**: Can work on triangular or quadrilateral mesh. Native support for **FESOM**, **ICON**, **NEMO**, and regular **Longitude-Latitude** meshes.
*   **🌐 Reduced Gaussian grids**: native support for ECMWF's classical (**N320**) and octahedral (**O96**) reduced Gaussian grids used by IFS, ERA5 and AIFS.
*   **♺ Element and Node Filtering**: Supports filtering on both mesh nodes and elements (triangles) natively for triangular meshes, automatically adjusting based on input data size.
*   **Variable scale filtering**: Filter size can be set individually for each mesh node 
*   **🚀 GPU Accelerated**: optimized for Nvidia GPUs and Apple Silicon using [JAX](https://jax.readthedocs.io/) for massive performance gains.
*   **⚡ Efficient**: Optimised for handling even the largest datasets.
*   **💾 Smart Caching**: Save and reload computed filter matrices to avoid redundant calculations.

---

## 📦 Installation

### 1. Standard Installation (CPU)
If you do not require GPU acceleration, install directly from GitHub:

```bash
python -m pip install "implicit_filter[cpu] @ git+https://github.com/FESOM/implicit_filter.git"
```
2. GPU Installation (Recommended)

For optimal performance, usage of an Nvidia GPU or Apple Silicon is highly recommended. The package uses JAX for hardware acceleration.

### For CUDA 12.x
```bash
python -m pip install "implicit_filter[cuda12] @ git+https://github.com/FESOM/implicit_filter.git"
```

### For AMD GPUs (ROCm)
```bash
python -m pip install "implicit_filter[rocm] @ git+https://github.com/FESOM/implicit_filter.git"
```

This pulls JAX's ROCm plugin, which targets ROCm 6.x; a matching ROCm installation has to be present on the machine already.

### For Google TPUs
```bash
python -m pip install "implicit_filter[tpu] @ git+https://github.com/FESOM/implicit_filter.git"
```

### For Apple Silicon (M1/M2/M3)
```bash
python -m pip install "implicit_filter[apple] @ git+https://github.com/FESOM/implicit_filter.git"
```

This installs Apple's `jax-metal` plugin, which JAX uses for Metal GPUs; it is experimental and macOS-only. `jax-metal` releases lag JAX by a long way, so it may not support the latest `jax`; if the plugin refuses to load, pin `jax` to a version it supports.

## 🚀 Quick Start

Here is a complete example of how to load a FESOM mesh, prepare the filter, and apply it to Scalar data (e.g., SSH).

```python
from implicit_filter import FesomFilter

# 1. Load Data
path = "/path/to/your/data/"
mesh_path = path + "fesom.mesh.diag.nc"
data = xr.open_dataset(path + "ssh.nc")
unfiltered_data = data['ssh'].values[0, :]

# 2. Initialize Filter
flter = FesomFilter()

# 2a. Select the backend FIRST. Importing implicit_filter pins JAX to CPU, and
# JAX fixes its platform when the first array is created, so set_backend has to
# come before prepare_from_file (and before any other compute) in the process:
flter.set_backend("gpu")

# 2b. Build the filter
flter.prepare_from_file(mesh_path)
# Note: If JAX prints a warning about GPU unavailability ignore it.

# 3. Caching (Optional but Recommended)
# Save auxiliary arrays to disk. These are mesh-specific and 
# only need to be computed once.
flter.save_to_file("filter_cache")

# ... later you can load it directly:
# flter = FesomFilter.load_from_file("filter_cache.npz")

# 4. Define Filter Parameters
distance = 100  # Target filter size (e.g., km)

# 5. Apply Filter
filtered_data = flter.compute(1, 2*math.pi / distance, unfiltered_data)
```

Select between CPU and GPU with the `set_backend` method. It has to be called
before the first `prepare` or `compute` call in the process, because JAX fixes
its platform when the first array is created:

```python
flter.set_backend("cpu")
# or 
flter.set_backend("gpu")

flter.get_backend()   # -> "cpu" or "gpu"
```

> **Note on GPU selection.** Importing `implicit_filter` sets JAX's platform to
> CPU for the whole process. `set_backend("gpu")` is therefore required to use a
> GPU, and it also affects any other JAX code running in the same interpreter.
> JAX fixes its platform on first use, so call `set_backend` **before** the
> first compute (or array-creating call) in the process.

> **Note on precision.** Importing `implicit_filter` also switches JAX into
> 64-bit mode (`jax.config.update("jax_enable_x64", True)`) for the whole
> process. Filter state and results are therefore `float64`, where JAX's own
> default is `float32`. The setting is global: any other JAX code in the same
> interpreter gets `float64` defaults too (`jnp.ones(3).dtype` becomes
> `float64`), which costs memory and speed in code written for `float32`. The
> filter needs it — the operator's condition number grows as `(L/dx)^{2n}` and
> leaves float32 far behind — so turning it back off is not supported.

## 🚀 V-cycle preconditioner

Biharmonic filters (`n=2`) at large filter-scale-to-resolution ratios can make
the default Jacobi-CG solver need thousands of iterations or fail outright. The
opt-in multigrid V-cycle preconditioner solves these stiff configurations in
tens of iterations:

```sh
pip install "implicit_filter[vcycle]"   # pyamg + scipy, setup-time only
```

```python
flter.set_preconditioner("vcycle")      # 'jacobi' (default) | 'none' | 'vcycle'
filtered = flter.compute(2, 2*math.pi / distance, data)
```

Works identically on CPU and GPU; requires a spatially uniform filter scale.
See the [documentation](https://fesom.github.io/implicit_filter/) for how it
works, tuning knobs, limitations (e.g. curvilinear NEMO `mesh_mask` grids are
rejected; stretched regular lat-lon grids are supported), and measured
before/after benchmarks.

## 🔺 Filtering on elements

For triangular meshes the filter works on **nodes** (vertices) or on
**elements** (triangle centres). The element operator is not built by default —
ask for it with `filter_elements=True`:

```python
flter = FesomFilter()
flter.prepare_from_file(mesh_path, filter_elements=True)

filtered_nodes    = flter.compute(1, 2*math.pi / distance, data_on_nodes)
filtered_elements = flter.compute(1, 2*math.pi / distance, data_on_elements)
```

Whether the data sits on nodes or elements is inferred from its length. On a
mesh with as many elements as nodes that is ambiguous, so you can say it
outright:

```python
filtered = flter.compute(1, k, data, on="elements")   # or on="nodes"
```

### Element weighting scheme

The element Laplacian supports two weightings, selected at `prepare` time:

| `elem_weights` | Behaviour |
| --- | --- |
| `"equilateral"` (default) | Fixed `sqrt(3)/area` coefficient — the finite-volume weight for an equilateral triangle. Depends only on cell area, ignores cell shape. This is the long-standing behaviour and is the default so existing results stay reproducible. |
| `"geometric"` | Uses the per-edge weights computed from the actual mesh geometry. Identical to the above on an equilateral mesh; more accurate on anisotropic or strongly graded meshes. |

```python
flter.prepare_from_file(mesh_path, filter_elements=True, elem_weights="geometric")
```

For advanced performance, you can also warm-start the iterative solver if you have a good initial guess (e.g., from a previous time step or similar filtering scale):
```python
filtered_data = flter.compute(1, 2*math.pi / distance, unfiltered_data, x0=previous_guess)
```

## 🌐 Reduced Gaussian grids (ECMWF, ERA5, AIFS)

ECMWF products live on *reduced Gaussian grids*: classical `N` grids such as
**N320** (native ERA5) and octahedral `O` grids such as **O96** (Anemoi ERA5,
AIFS training). `ReducedGaussianFilter` triangulates the grid points on the
sphere and filters them as mesh nodes, with spherical geometry that needs no
special treatment at the poles or the date line:

```python
from implicit_filter import ReducedGaussianFilter

flter = ReducedGaussianFilter()
flter.prepare_from_grid("N320")             # or "O96", "N128", ...
filtered = flter.compute(1, 2*math.pi / distance, data)   # data: the GRIB `values` array
```

`data` is the 1-D `values` array of a GRIB message (or of the ERA5 zarr
stores): rows from north to south, west to east from 0° longitude. The filter
can also be built from the coordinates stored with the data, which works for
any set of points covering the sphere:

```python
ds = xr.open_dataset("era5.grib", engine="cfgrib")
flter.prepare_from_data_array(ds)                 # 1-D latitude/longitude coordinates
flter.prepare_from_points(lat, lon)                # plain arrays in degrees
flter.prepare_from_points(lat, lon, mask=~np.isnan(sst))   # skip land points
# fill NaN first: masked points are returned unchanged
filtered = flter.compute(1, 2*math.pi / distance, np.where(np.isnan(sst), 0.0, sst))
```

Everything else — velocity filtering, spectra, caching with `save_to_file`,
`set_backend("gpu")`, the V-cycle preconditioner — works as for any
triangular mesh. Element filtering is not available (the data is nodal).
`prepare` takes about 20 s and the cache about 260 MB per N320-sized grid
(540 k points); larger grids scale accordingly.
See [`examples/reduced_gaussian_grid_example.ipynb`](examples/reduced_gaussian_grid_example.ipynb)
for a walkthrough on public ERA5 data (O96 and N320).

## 🤝 Support & Feature Requests

**Missing a feature? Using a model grid that isn't supported yet?**

I am actively developing this package and am always happy to help! If you are interested in using Implicit Filter but find something missing:

1.  **Open an Issue**: Please describe your use case or the mesh you are using.
2.  **Get Quick Support**: Adding support for new meshes or implementing specific features can often be done quickly.

Don't hesitate to reach out—feedback and new use cases are highly appreciated! 🚀


## 🛠️ Developer Setup

Requires Python 3.10+.

```bash
git clone https://github.com/FESOM/implicit_filter
cd implicit_filter

# Install in editable mode
pip install -e .
```

## 📄 Citation

If you use this package in your research, please cite: https://doi.org/10.5194/gmd-18-6541-2025

