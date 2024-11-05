
![Logo](logo.png)

# Implicit Filter Python Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10907365.svg)](https://doi.org/10.5281/zenodo.10907365)

The Implicit Filter Python Package provides a collection of functions and classes for filtering data using implicit filtering techniques.
Originally FESOM, and now ICON models are supported.

For optimal performance usage of Nvidia GPU is highly recommended.

## ICON Version Change Notes
_Aaron Wienkers, 2024_

1. Built-in support for the `ICON` grid
    - Uses new JAX routines for vertex- and cell-centred transformations
2. Support for spatially-varying length-scale filters, e.g. $\Delta \propto \mathcal{L}_\mathrm{Ro}$
3. Implements _Neumann_ Boundary Conditions in a conservative fashion
    - This is necessary when filtering deeper layers in the ocean when the sea-floor boundaries change
    - These BCs ensure no artificial mass/energy/tracer leakage through the boundaries, and which otherwise produces halos in the high-pass fields
4. Adds MPI support for multi-GPU machines
    - Batches GPU computations (in time & wavenumber space) to avoid running out of memory
5. Improves convergence properties of 2nd Order filter using a fine-tuned Algebriac Multigrid Solver (AMGX)
    - CG Convergence of the 2nd Order filter with the original block-wise structure was either slow or (for larger filter length-scales $\sim 100$ km) divergent
    - AMGX Solver Converges faster on 2nd Order Filter compared to `cupy` CG implementation of 1st Order Filter, even up to $\Delta \approx 10^4$ km
    - N.B.: Using 2nd Order filter is necessary for many higher-order filtered statistics and when a sharp(er) filter cutoff is required

Other Notes:
- AMGX Filter cannot handle un-masked NaNs within the domain (e.g. if large values are dropped)
- Batch processing in time (or spectrally) requires that dimension to be in a single chunk 

## Tutorial for ICON Data

See the attached Jupyter Notebook, `./examples/intro_icon_filtering.ipynb` for a short introduction to using specifically `implicit_filter_ICON` with ICON model data.


## Dependencies

**Base:** 
- NumPy & ScipPy
- JAX
- CuPy — _Most success when installing from `pip`_
- AMGX — _Build from [source](https://github.com/NVIDIA/AMGX/tree/main)_
- pyAMGX — _Define `AMGX_DIR` (location above), and then `pip install pyamgx`_
    - Alternatively, point to build on DKRZ Levante `AMGX_DIR=/home/b/b382615/opt/AMGX`

**IO:** 
- Xarray
- Dask

**Visualization:** 
- Matplotlib




## Installation
Easiest:
```shell
pip install git+https://github.com/wienkers/implicit_filter_ICON.git
```

Development Installation: 
```shell
source ./path/to/enviroment/of/your/choice
git clone https://github.com/wienkers/implicit_filter_ICON.git

cd implicit_filter
pip install -e .
```


