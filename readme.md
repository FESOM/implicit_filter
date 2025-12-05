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
*   **Variable scale filtering**: Filter size can be set individually for each mesh node 
*   **🚀 GPU Accelerated**: optimized for Nvidia GPUs using [CuPy](https://cupy.dev/) for massive performance gains.
* **Variable scale filering**
*   **⚡ Efficient**: Optimised for handling even the largest datasets.
*   **💾 Smart Caching**: Save and reload computed filter matrices to avoid redundant calculations.

---

## 📦 Installation

### 1. Standard Installation (CPU)
If you do not require GPU acceleration, install directly from GitHub:

```bash
python -m pip install git+https://github.com/FESOM/implicit_filter.git
```
2. GPU Installation (Recommended)

For optimal performance, usage of an Nvidia GPU is highly recommended. You must install cupy.

Option A: Automatic (via extras)
Install the package with the tag matching your CUDA version (check via nvidia-smi):

# For CUDA 11.x
```bash
python -m pip install "implicit_filter[gpu_c11] @ git+https://github.com/FESOM/implicit_filter.git"
```

# For CUDA 12.x
```bash
python -m pip install "implicit_filter[gpu_c12] @ git+https://github.com/FESOM/implicit_filter.git"
```

Option B: Manual
Install CuPy separately before installing the package:

```bash
pip install cupy-cuda12x  # Adjust for your CUDA version
python -m pip install git+https://github.com/FESOM/implicit_filter.git
```

🚀 Quick Start

Here is a complete example of how to load a FESOM mesh, prepare the filter, and apply it to Scalar data (e.g., SSH).

```python
from implicit_filter import FesomFilter

# 1. Load Data
path = "/path/to/your/data/"
mesh_path = path + "fesom.mesh.diag.nc"
data = xr.open_dataset(path + "ssh.nc")
unfiltered_data = data['ssh'].values[0, :]

# 2. Initialize Filter
# You can enable GPU immediately during initialization or later at any point
flter = FesomFilter()
flter.prepare_from_file(mesh_path, gpu=True) 
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

You can switch between CPU and GPU at runtime without restarting:

```python
flter.set_backend("cpu")
# or 
flter.set_backend("gpu")
```

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

