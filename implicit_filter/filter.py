from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import numpy as np


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
        Compute the filtered velocity data using a specified filter size.

        It performs implicit interpolates it from elements to nodes. Currently, filtering directly on elements
        is not supported.

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

    @abstractmethod
    def many_compute(self, n: int, k: float, data: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Computes multiple inputs, which are scalar data

        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : float
            Wavelength of the filter.

        data : Union[np.ndarray, List[np.ndarray]]
            It can be either a list of 1D NumPy arrays to be processed or
            a 2D NumPy array which 2nd dimension will be iterated over.

        Returns:
        --------
        List[np.ndarray]:
            List containing NumPy arrays with filtered data
        """
        pass

    def many_compute_velocity(self, n: int, k: float, ux: Union[np.ndarray, List[np.ndarray]],
                              vy: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes multiple velocity inputs

        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : float
            Wavelength of the filter.

        ux : np.ndarray
            Eastward velocity component to be filtered. It can be either a list of 1D NumPy arrays to be processed or
            a 2D NumPy array which 2nd dimension will be iterated over.

        uy : np.ndarray
            Northwards velocity component to be filtered. It can be either a list of 1D NumPy arrays to be processed or
            a 2D NumPy array which 2nd dimension will be iterated over.

        Returns:
        --------
        Tuple[List[np.ndarray], List[np.ndarray]]:
            Tuple containing lists containing NumPy arrays with filtered data
        """
        pass

    def save_to_file(self, file: str):
        """Save auxiliary arrays to file, as they are mesh specific"""
        np.savez(file, **vars(self))

    @classmethod
    def load_from_file(cls, file: str):
        """Load auxiliary arrays from file"""
        return cls(**dict(np.load(file)))
