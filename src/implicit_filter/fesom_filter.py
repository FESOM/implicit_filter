import math

import xarray as xr
import numpy as np
from .triangular_filter import TriangularFilter


class FesomFilter(TriangularFilter):
    """
    Filter implementation specialized for FESOM ocean model meshes.

    This class extends the TriangularFilter to work natively with FESOM
    mesh files and data structures. It provides convenience methods for
    loading mesh configurations directly from FESOM output files.

    Parameters
    ----------
    See TriangularFilter for inherited parameters.

    Notes
    -----
    Supports FESOM mesh files containing either 'elements' or 'face_nodes'
    variables to define element connectivity. Automatically handles FESOM's
    1-based indexing conversion to 0-based indexing for Python.
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
        Configure filter using a FESOM mesh file.

        Parameters
        ----------
        file : str
            Path to FESOM mesh file (NetCDF format).
        meshtype : str, optional
            Mesh type identifier:
            - 'm': Metric tensor formulation (default)
            - 'r': Radial formulation for spherical coordinates
        cartesian : bool, optional
            True for Cartesian coordinates, False for spherical (default).
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        metric : bool, optional
            True to include metric terms in operator (default: False).
        mask : np.ndarray, optional
            Element land mask where True indicates land (default: None).
        gpu : bool, optional
            True to enable GPU acceleration (default: False).

        Notes
        -----
        This method automatically:
        1. Reads the mesh file using xarray
        2. Extracts node coordinates and element connectivity
        3. Converts FESOM's 1-based indexing to 0-based indexing
        4. Configures the filter using the parent class prepare method
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
        """
        Configure filter using an xarray Dataset containing FESOM mesh data.

        Parameters
        ----------
        mesh : xr.Dataset
            xarray Dataset containing FESOM mesh variables.
        meshtype : str, optional
            Mesh type identifier (default: 'm').
        cartesian : bool, optional
            Coordinate system flag (default: False).
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        metric : bool, optional
            Metric terms inclusion flag (default: False).
        mask : np.ndarray, optional
            Element land mask (default: None).
        gpu : bool, optional
            GPU acceleration flag (default: False).

        Raises
        ------
        RuntimeError
            If mesh doesn't contain recognizable element connectivity data.

        Notes
        -----
        Looks for element connectivity in variables named:
        - 'elements'
        - 'face_nodes'
        - 'elem'
        """
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
