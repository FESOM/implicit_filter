from abc import ABC, abstractmethod
from typing import Tuple, Iterable

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
    def set_backend(self, backend: str):
        """
        Set the computational backend for filter operations.

        Parameters
        ----------
        backend : str
            Name of the backend to use (e.g., 'cpu', 'gpu').
        """
        pass

    @abstractmethod
    def get_backend(self) -> str:
        """
        Get the current computational backend.

        Returns
        -------
        str
            Name of the active backend.
        """
        pass

    @abstractmethod
    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        """
        Compute the filtered data using a specified filter size.
        Data must be placed on mesh nodes

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
    def compute_velocity(
        self, n: int, k: float, ux: np.ndarray, vy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the filtered velocity data using a specified filter size.
        Data must be placed on mesh nodes

        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : float
            Wavelength of the filter.

        ux : np.ndarray
            NumPy array containing eastward velocity component to be filtered.

        vy : np.ndarray
            NumPy array containing northwards velocity component to be filtered.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            Tuple containing NumPy arrays with filtered data ux and uy velocities on mesh nodes.
        """
        pass

    @abstractmethod
    def compute_spectra_scalar(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes power spectra for given wavelengths.
        Data must be placed on mesh nodes

        For details refer to https://arxiv.org/abs/2404.07398
        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : Iterable | np.ndarray
            List of wavelengths to be filtered.

        data : np.ndarray
            NumPy array containing data to be filtered.

        mask : np.ndarray | None
            Mask applied to data while computing spectra.
            True means selected data won't be used for computing spectra.
            This mask won't be used during filtering.

        Returns:
        --------
        np.ndarray:
            Array containing power spectra for given wavelengths.
        """
        pass

    @abstractmethod
    def compute_spectra_velocity(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes power spectra for given wavelengths.
        Data must be placed on mesh nodes

        For details refer to https://arxiv.org/abs/2404.07398
        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : Iterable | np.ndarray
            List of wavelengths to be filtered.

        ux : np.ndarray
            NumPy array containing an eastward velocity component to be filtered.

        vy : np.ndarray
            NumPy array containing a northwards velocity component to be filtered.

        mask : np.ndarray | None
            Mask applied to data while computing spectra.
            True means selected data won't be used for computing spectra.
            This mask won't be used during filtering.

        Returns:
        --------
        np.ndarray:
            Array containing power spectra for given wavelengths.
        """
        pass

    def __getstate__(self):
        # Only include names that start with '_'
        return {k: v for k, v in vars(self).items() if k.startswith("_")}

    def save_to_file(self, file: str):
        """
        Persist internal state to NPZ file.

        Parameters
        ----------
        file : str
            Output file path (.npz extension recommended)
        """
        np.savez(file, **self.__getstate__())

    @classmethod
    def load_from_file(cls, file: str):
        """
        Instantiate filter from saved state file.

        Parameters
        ----------
        file : str
            Input file path created by save_to_file()

        Returns
        -------
        Filter
            Reconstructed filter instance with restored state
        """
        return cls(**dict(np.load(file)))
