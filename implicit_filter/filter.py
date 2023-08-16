from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import xarray as xr


class Filter(ABC):
    """
    Abstract base class for filters
    """
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @abstractmethod
    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        """
        Compute the filtered data using a specified filter size.

        Parameters:
        ------------
        n : int
            Order of filter, one is recommended

        k : float
            Wavelength of the filter.

        data : np.ndarray
            NumPy array containing data to be filtered.

        Returns:
        --------
        np.ndarray
            NumPy array with filtered data.
        """
        pass

    @abstractmethod
    def compute_velocity(self, n: int, k: float, ux: np.ndarray, uy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the filtered data using a specified filter size.

        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : float
            Wavelength of the filter.

        ux : np.ndarray
            NumPy array containing eastward velocity component to be filtered.

        uy : np.ndarray
            NumPy array containing northwards velocity component to be filtered.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            Tuple containing NumPy arrays with filtered data ux and uy velocities.
        """
        pass

    def save_to_file(self, file: str):
        """Save auxiliary arrays to file, as they are mesh specific"""
        np.savez(file, **vars(self))

    @classmethod
    def load_from_file(cls, file: str):
        """Load auxiliary arrays from file"""
        return cls(**dict(np.load(file)))

    @abstractmethod
    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
                carthesian: bool, cyclic_length: float, full: bool):
        """
        Prepare the filter to be used with the given mesh.

        Parameters:
        -----------
        n2d : int
            The total number of nodes in the mesh.

        e2d : int
            The total number of triangles (elements) in the mesh.

        tri : np.ndarray
            A 2D NumPy array representing the connectivity of triangles (elements) in the mesh.
            Each row contains the indices of the three nodes that form a triangle.

        xcoord : np.ndarray
            A 1D NumPy array containing the x-coordinates of nodes in the mesh.

        ycoord : np.ndarray
            A 1D NumPy array containing the y-coordinates of nodes in the mesh.

        meshtype : str
        Mesh type, either 'm' (metric) or 'r' (radial).

        carthesian : bool
            Boolean indicating whether the mesh is in Cartesian coordinates.

        cyclic_length : float
            The length of the cyclic boundary if the mesh is cyclic (for 'r' meshtype).

        full : bool, optional
            A flag indicating whether to use the calculation including metric factors (True) or not (False).
            Default is False.
        """
        pass

    def prepare_from_file(self, file: str, meshtype: str, carthesian: bool, cyclic_length: float, metric: bool = False):
        """
        Prepare the filter to be used with a mesh provided in the given file path.

        Parameters:
        -----------
        file : str
            Path to the FESOM mesh file.

        meshtype : str
        Mesh type, either 'm' (metric) or 'r' (radial).

        carthesian : bool
            Boolean indicating whether the mesh is in Cartesian coordinates.

        cyclic_length : float
            The length of the cyclic boundary if the mesh is cyclic (for 'r' meshtype).

        metric : bool, optional
            A flag indicating whether to use the calculation including metric terms (True) or not (False).
            Default is False.
        """
        mesh = xr.open_dataset(file)
        xcoord = mesh['lon'].values
        ycoord = mesh['lat'].values
        tri = mesh['elements'].values.T - 1

        self.prepare(len(xcoord), len(tri[:, 1]), tri, xcoord, ycoord, meshtype, carthesian, cyclic_length, metric)
