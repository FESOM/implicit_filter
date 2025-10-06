
![Logo](logo.png)

# Implicit Filter Python Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg)](https://doi.org/10.5281/zenodo.10907365)
[![arXiv](https://img.shields.io/badge/arXiv-2404.07398-b31b1b.svg)](https://arxiv.org/abs/2404.07398)

The Implicit Filter Python Package provides a collection of classes for filtering data using implicit filtering techniques.
Currently FESOM, ICON, NEMO and longitude-latitude meshes are supported.

For optimal performance usage of Nvidia GPU is highly recommended.

For details of the implementation please read out paper on [GMD](https://gmd.copernicus.org/articles/18/6541/2025/)

For full mathematical formulation of the implicit filter please refer to [this paper](http://dx.doi.org/10.1029/2023MS003946)
## Installation 

### End-user installation

```shell
# create env with cupy (e.g. from Conda)
# activate env
python -m pip install git+https://github.com/FESOM/implicit_filter.git
```

If one wants to use GPU it's necessary to install [cupy](https://cupy.dev/). It can be installed with the pachage by adding optional dependency. Check your Nvidia driver version using `nvidia-smi` and install CuPy version matching your drivers. You can install it separately 
```shell
pip install cupy-cuda11x # CUDA 11 
pip install cupy-cuda12x # CUDA 12 or newer
```
It can be also installed with the package (example for CUDA 12)
```shell
python -m pip install "implicit_filter[gpu_c12] @ git+https://github.com/FESOM/implicit_filter.git"
```

# Tutorial

This is a basic example, for more advanced usecases please look into the examples. 

Lets start with loading FESOM mesh file and data that we want to filter

This is a basic example with only scalar data.
```python
import xarray as xr

path = "your path"
mesh = xr.open_dataset(path + "fesom.mesh.diag.nc")
data = xr.open_dataset(path + "ssh.nc")

unfiltered = data['ssh'].values[0, :]
```

Now create filter.

The easiest way to do it is by using mesh path. Alternatively, you can set arrays by yourself, but this is shown in notebooks in examples.

```python
from implicit_filter import FesomFilter 

flter = FesomFilter()
flter.prepare_from_file(path + "fesom.mesh.diag.nc")
```
JAX warning might appear about GPU not being available, but it should be ignored. 


To use GPU you need to either pass this option in filter preparation or set backend later. Backends can be switched during runtime without causing any issues. 
```python
from implicit_filter import FesomFilter 

flter = FesomFilter()
flter.prepare_from_file(path + "fesom.mesh.diag.nc", gpu=True)

flter.set_backend("gpu")
```

It is highly recommended to save filter's auxiliary arrays as it can take a significant amount of time to compute them.
Auxiliary arrays are specific to each mesh, so they only need to be computed once.

```python
flter.save_to_file("filter_cash")
```

Later filter can be created based on this file

```python
flter = FesomFilter.load_from_file("filter_cash.npz")
```

Now you can define wavenumber of the filter in two ways.

Using function from the package:

```python
from implicit_filter import convert_to_wavenumbers

distance = 100 # filter size 
dxm = 5 # mesh resolution 
# Units has to be consistent among mesh file and distance and dxm

k = convert_to_wavenumbers(distance, dxm)
```

or manually:

```python
wavelength = 70
Kc = wavelength * dxm
k = 2 * math.pi / Kc
```

Finally you can filter your data

```python
filtered = flter.compute(1, k, unfiltered)
```

### Developer installation

Currently Python version has to be 3.10 or newer. 
```shell
source ./path/to/enviroment/of/your/choice
git clone https://github.com/FESOM/implicit_filter

cd implicit_filter
# CPU only installation
pip install -e .
```

## Dependencies

**CPU only:** NumPy, ScipPy, JAX

**GPU accelerated:** NumPy, CuPy, JAX

**Visualization:** Matplotlib

**IO:** Xarray
