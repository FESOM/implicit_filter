
![Logo](logo.png)

# Implicit Filter Python Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg)](https://doi.org/10.5281/zenodo.10907365)

The Implicit Filter Python Package provides a collection of functions and classes for filtering data using implicit filtering techniques.
Currently FESOM and now ICON models are supported.

For optimal performance usage of Nvidia GPU is highly recommended.


## Dependencies

**CPU only:** NumPy, ScipPy, JAX

**GPU accelerated:** NumPy, CuPy, JAX, AMGX, pyAMGX

**Visualization:** Matplotlib

**IO:** Xarray

## ICON Version Change Notes

Other Notes:
- AMGX Filter does not handle nans within the domain well (e.g. if large values are masked)
- Batch processing in time (or spectrally) requires that dimension to be in a single chunk 


## Installation 
Currently Python version 3.10 and 3.9 are supported. Using newer version can enabled 
```shell
source ./path/to/enviroment/of/your/choice
git clone https://github.com/FESOM/implicit_filter

cd implicit_filter
# CPU only installation
pip install -e .
# GPU installation
pip install -e .[gpu]
```
### Known issues
Installing CuPy can cause an error, in case this happens try installing it manually:

```shell
pip install cupy
```

In case it also doesn't work, check your Nvidia driver version using `nvidia-smi` and install 
CuPy version matching your drivers.

# Tutorial

Lets start with loading FESOM mesh file and data that we want to filter

This is basic example with only scalar data.
```python
import xarray as xr

path = "your path"
mesh = xr.open_dataset(path + "fesom.mesh.diag.nc")
data = xr.open_dataset(path + "ssh.nc")

unfiltered = data['ssh'].values[0, :]
```

Now we can create filter.

The easiest way to do it is by using mesh file path

```python
from implicit_filter import CuPyFilter 
# if you don't have GPU use JaxFilter instead

flter = CuPyFilter()
flter.prepare_from_file(path + "fesom.mesh.diag.nc")
```
JAX warning might appear about GPU not being available, but it should be ignored. 

If you don't have GPU support enabled importing CuPyFilter will cause an import error.

Alternatively you can set arrays by yourself, but this is shown in notebooks in examples.

It is highly recommended to save filter's auxiliary arrays as it can take significant amount of time to compute them.
Auxiliary arrays are specific to each mesh, so they only need to be computed once.

```python
flter.save_to_file("filter_cash")
```

Later filter can be created based on this file

```python
flter = CuPyFilter.load_from_file("filter_cash.npz")
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

