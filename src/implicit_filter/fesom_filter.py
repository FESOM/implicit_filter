import math

import xarray as xr
import numpy as np
from .triangular_filter import TriangularFilter


class FesomFilter(TriangularFilter):
    """
    Class for filtering data based on FESOM triangular meshes.
    """

    def prepare_from_file(
        self,
        file: str,
        meshtype: str = "m",
        cartesian: bool = False,
        cyclic_length: float = 360.0 * math.pi / 180.0,
        metric: bool = False,
        mask: np.ndarray = None,
        gpu: bool = False,
    ):
        """
        Prepare the filter to be used with a mesh provided in the given file path.

        Parameters:
        -----------
        file : str
            Path to the FESOM mesh file.

        meshtype : str
        Mesh type, either 'm' (metric) or 'r' (radial).
        Default is metric.

        carthesian : bool
            Boolean indicating whether the mesh is in Cartesian coordinates. Default is False

        cyclic_length : float
            The length of the cyclic boundary if the mesh is cyclic (for 'r' meshtype). Default is 360 * pi / 180

        metric : bool, optional
            A flag indicating whether to use the calculation including metric terms (True) or not (False).
            Default is False.

        """
        mesh = xr.open_dataset(file)
        self.prepare_from_data_array(
            mesh, meshtype, cartesian, cyclic_length, metric, mask, gpu
        )

    def prepare_from_data_array(
        self,
        mesh: xr.DataArray,
        meshtype: str = "m",
        cartesian: bool = False,
        cyclic_length: float = 360.0 * math.pi / 180.0,
        metric: bool = False,
        mask: np.ndarray = None,
        gpu: bool = False,
    ):
        xcoord = mesh["lon"].values
        ycoord = mesh["lat"].values

        keys = mesh.keys()
        if "elements" in keys:
            tri = mesh["elements"].values.T - 1
        elif "face_nodes" in keys:
            tri = mesh["face_nodes"].values.T - 1
        elif "elem" in keys:
            tri = mesh["elem"].values.T - 1
        else:
            raise RuntimeError(
                "In FESOM mesh file triangulation data was not found. It should be either named as elements or face_nodes"
            )

        self.prepare(
            len(xcoord),
            len(tri[:, 1]),
            tri,
            xcoord,
            ycoord,
            meshtype,
            cartesian,
            cyclic_length,
            metric,
            mask,
            gpu,
        )
