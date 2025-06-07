from .filter import Filter
from .triangular_filter import TriangularFilter
from .fesom_filter import FesomFilter
# from icon_filter import IconFilter
# from .nemo_filter import NemoNumpyFilter
# from .latlon_filter import LatLonNumpyFilter
from .utils._auxiliary import make_tri, convert_to_wavenumbers
from .utils.conversion_tools import transform_velocity_to_nodes, transform_scalar_to_nodes

# If CuPy isn't installed, this classes won't be imported
# try:
#     from .cupy_filter import CuPyFilter
#     from .nemo_cupy_filter import NemoCupyFilter
#     from .latlon_cupy_filter import LatLonCupyFilter
# except ModuleNotFoundError:
#     pass
#
# # The same for pyAMGX
# try:
#     from .amgx_filter import AMGXFilter
# except ModuleNotFoundError:
#     pass