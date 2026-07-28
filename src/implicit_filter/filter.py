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

    def get_backend(self) -> str:
        """
        Report the currently selected backend as ``'cpu'`` or ``'gpu'``.

        The returned value is always accepted by :meth:`set_backend`, so the
        two round-trip. JAX stores a platform *priority list* (e.g.
        ``'gpu,cpu'``); this reports the platform that would actually be
        preferred rather than that raw string.
        """
        import jax
        platforms = jax.config.jax_platforms
        if not platforms:
            return "gpu"
        return "cpu" if platforms.split(",")[0].strip().lower() == "cpu" else "gpu"

    def set_backend(self, backend: str):
        """
        Force JAX to use a specific backend (e.g. 'cpu' or 'gpu').
        This is useful if you want to run on CPU while the GPU is busy.
        """
        import jax
        if backend.lower() == "cpu":
            jax.config.update("jax_platforms", "cpu")
        else:
            jax.config.update("jax_platforms", "gpu,cpu")

    @abstractmethod
    def compute(self, n: int, k: float | np.ndarray, data: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the filtered data using a specified filter size.
        Data must be placed on mesh nodes

        Parameters:
        ------------
        n : int
            Order of filter, one is recommended

        k : float | np.ndarray
            Wavelength of the filter.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data

        data : np.ndarray
            NumPy array containing data to be filtered.

        x0 : np.ndarray | None, optional
            Optional initial guess for the CG solver. Can be provided to accelerate convergence.
            Defaults to None.

        Returns:
        --------
        np.ndarray
            NumPy array with filtered data.
        """
        pass

    @abstractmethod
    def compute_velocity(
        self, n: int, k: float | np.ndarray, ux: np.ndarray, vy: np.ndarray, ux0: np.ndarray | None = None, vy0: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the filtered velocity data using a specified filter size.
        Data must be placed on mesh nodes

        Parameters:
        -----------
        n : int
            Order of filter, one is recommended

        k : float | np.ndarray
            Wavelength of the filter.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data

        ux : np.ndarray
            NumPy array containing eastward velocity component to be filtered.

        vy : np.ndarray
            NumPy array containing northwards velocity component to be filtered.

        ux0 : np.ndarray | None, optional
            Optional initial guess for the x-component CG solver. Defaults to None.
            
        vy0 : np.ndarray | None, optional
            Optional initial guess for the y-component CG solver. Defaults to None.

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

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

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

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

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

        Notes
        -----
        Attributes that were never populated (e.g. the element operators when
        the filter was prepared with ``filter_elements=False``) are omitted
        rather than written as ``None``. NPZ cannot store ``None`` without
        pickling, and the resulting file would not be readable by
        :meth:`load_from_file`, which loads with ``allow_pickle=False``.
        Omitted attributes are restored as ``None`` on load.
        """
        state = {k: v for k, v in self.__getstate__().items() if v is not None}
        np.savez(file, **state)

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
