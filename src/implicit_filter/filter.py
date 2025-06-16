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
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
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
    def compute_velocity(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def compute_spectra_scalar(self, n: int, k: Iterable | np.ndarray, data: np.ndarray,
                               mask: np.ndarray | None = None) -> np.ndarray:
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
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(data.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]

        spectra[-1] = np.sum(selected_area * (np.square(data))[not_mask]) / np.sum(selected_area)

        for i in range(nr):
            ttu = self.compute(n, k[i], data)
            ttu -= data

            ttu[mask] = 0.0
            spectra[i] = np.sum(selected_area * (np.square(ttu))[not_mask]) / np.sum(selected_area)

        return spectra

    def compute_spectra_velocity(self, n: int, k: Iterable | np.ndarray, ux: np.ndarray, vy: np.ndarray,
                                 mask: np.ndarray | None = None) -> np.ndarray:
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
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(ux.shape, dtype=bool)

        unod = ux
        vnod = vy

        not_mask = ~mask
        selected_area = self._area[not_mask]
        spectra[-1] = np.sum(selected_area * (np.square(unod) + np.square(vnod))[not_mask]) / np.sum(selected_area)

        for i in range(nr):
            ttu = self.compute(n, k[i], unod)
            ttv = self.compute(n, k[i], vnod)

            ttu -= unod
            ttv -= vnod

            ttu[mask] = 0.0
            ttv[mask] = 0.0

            spectra[i] = np.sum(selected_area * (np.square(ttu) + np.square(ttv))[not_mask]) / np.sum(selected_area)

        return spectra

    def save_to_file(self, file: str):
        """Save auxiliary arrays to file, as they're mesh-specific"""
        np.savez(file, **vars(self))

    @classmethod
    def load_from_file(cls, file: str):
        """Load auxiliary arrays from a file"""
        return cls(**dict(np.load(file)))


