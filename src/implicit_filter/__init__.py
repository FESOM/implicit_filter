from .jax_filter import JaxFilter
from .numpy_filter import NumpyFilter
from .nemo_filter import NemoNumpyFilter
from .latlon_filter import LatLonNumpyFilter
from ._auxiliary import make_tri, convert_to_wavenumbers

# If CuPy isn't installed, this classes won't be imported
try:
    from .cupy_filter import CuPyFilter
    from .nemo_cupy_filter import NemoCupyFilter
    from .latlon_cupy_filter import LatLonCupyFilter
except ModuleNotFoundError:
    pass
