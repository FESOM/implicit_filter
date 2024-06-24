from .jax_filter import JaxFilter
from .numpy_filter import NumpyFilter
from .nemo_filter import NemoNumpyFilter
from ._auxiliary import make_tri, convert_to_wavenumbers

# If CuPy is not installed this class won't be imported
try:
    from .cupy_filter import CuPyFilter
    from .nemo_cupy_filter import NemoCupyFilter
except ModuleNotFoundError:
    pass
