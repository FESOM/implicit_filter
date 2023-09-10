
![Logo](logo.png)

# Implicit Filter Python Package

The Implicit Filter Python Package provides a collection of functions and classes for filtering data using implicit filtering techniques.
Currently only FESOM files are supported.

For optimal performance usage of Nvidia GPU is highly recommended.


## Dependencies

**CPU only:** NumPy, ScipPy, JAX

**GPU accelerated:** NumPy, CuPy, JAX

**Visualization:** Matplotlib

**IO:** Xarray

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

