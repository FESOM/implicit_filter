import math
from typing import Tuple, Iterable

import numpy as np

from implicit_filter.utils._numpy_functions import (
    calculate_local_regular_neighbourhood,
    calculate_global_regular_neighbourhood,
)
from implicit_filter.filter import Filter
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    get_backend,
    transform_attribute,
)


class LatLonFilter(Filter):
    """
    Filter implementation for regular latitude-longitude grids.

    This class provides implicit filtering capabilities for data on structured
    lat-lon grids. It supports both Cartesian and spherical coordinate systems
    with configurable boundary conditions and land-sea masks.

    Parameters
    ----------
    See Filter class for inherited parameters.

    Attributes
    ----------
    _e2d : int
        Total number of grid points (nx * ny)
    _nx : int
        Number of longitude points
    _ny : int
        Number of latitude points
    _ss : np.ndarray
        Non-zero values of sparse filter matrix
    _ii : np.ndarray
        Row indices for sparse matrix entries
    _jj : np.ndarray
        Column indices for sparse matrix entries
    _area : np.ndarray
        Area associated with each grid cell
    _backend : str
        Computational backend ('cpu' or 'gpu')
    _mask_n : np.ndarray
        Boolean mask for valid grid points (False indicates land)
    """
    def __init__(self, *initial_data, **kwargs):
        super().__init__(initial_data, **kwargs)
        it = lambda ar: int(ar)
        ar = lambda ar: np.array(ar)
        st = lambda ar: str(ar)

        # Transform and initialize attributes with default values
        transform_attribute(self, "_e2d", it, 0)
        transform_attribute(self, "_nx", it, 0)
        transform_attribute(self, "_ny", it, 0)
        transform_attribute(self, "_ss", ar, None)
        transform_attribute(self, "_ii", ar, None)
        transform_attribute(self, "_jj", ar, None)
        transform_attribute(self, "_area", ar, None)
        transform_attribute(self, "_backend", st, "cpu")

        self.set_backend(self._backend)

    def prepare(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        cartesian: bool = False,
        local: bool = True,
        cyclic_length: float = 2 * math.pi,
        mask: np.ndarray | None = None,
        gpu: bool = False,
    ):
        """
        Configure filter for a latitude-longitude grid.

        Computes grid topology, geometric properties, and assembles the filter
        operator matrix. Must be called before any filtering operations.

        Parameters
        ----------
        latitude : np.ndarray
            Latitude values in degrees (1D array)
        longitude : np.ndarray
            Longitude values in degrees (1D array)
        cartesian : bool, optional
            True for Cartesian coordinates, False for spherical (default)
        local : bool, optional
            True for 4-point local neighborhood, False for 8-point global (default: True)
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        mask : np.ndarray, optional
            Land-sea mask where True indicates land (default: all ocean)
        gpu : bool, optional
            True to enable GPU acceleration (default: False)

        Notes
        -----
        - Land points are masked using Neumann boundary conditions
        """
        nx = len(longitude)
        ny = len(latitude)
        e2d = nx * ny
        self._e2d = e2d
        self._nx = nx
        self._ny = ny

        xcoord = np.zeros((nx, ny))
        ycoord = xcoord.copy()

        for i in range(nx):
            ycoord[i, :] = latitude

        for i in range(ny):
            xcoord[:, i] = longitude

        xcoord = np.reshape(xcoord, [nx * ny])
        ycoord = np.reshape(ycoord, [nx * ny])

        self._mask_n = (
            np.ones(self._e2d, dtype=bool) if mask is None else mask.flatten()
        )

        if local:
            ee_pos, nza = calculate_local_regular_neighbourhood(e2d, nx, ny)
        else:
            ee_pos, nza = calculate_global_regular_neighbourhood(e2d, nx, ny)

        rad = math.pi / 180.0

        if cartesian:
            Mt = np.ones(e2d)
        else:
            Mt = np.cos(np.sum(rad * ycoord[ee_pos], axis=0) / 4.0)

        hh = np.ones((4, e2d))  # Edge lengths
        hc = np.ones((4, e2d))  # Distance to next cell centers
        r_earth = 6400.0

        # Fill ee_pos, arrangement is W;N;E;S
        for i in range(e2d):
            if ee_pos[1, i] == i:
                hc[1, i] = rad * r_earth * (ycoord[i] - ycoord[ee_pos[3, i]])  # S
            else:
                hc[1, i] = rad * r_earth * (ycoord[ee_pos[1, i]] - ycoord[i])  # N

            if ee_pos[3, i] == i:
                hc[3, i] = rad * r_earth * (ycoord[ee_pos[1, i]] - ycoord[i])  # N
            else:
                hc[3, i] = rad * r_earth * (ycoord[i] - ycoord[ee_pos[3, i]])  # S

            if ee_pos[0, i] == i:
                hc[0, i] = rad * (xcoord[ee_pos[2, i]] - xcoord[i])  # E
            else:
                hc[0, i] = rad * (xcoord[i] - xcoord[ee_pos[0, i]])  # W

            if ee_pos[2, i] == i:
                hc[2, i] = rad * (xcoord[i] - xcoord[ee_pos[0, i]])  # W
            else:
                hc[2, i] = rad * (xcoord[ee_pos[2, i]] - xcoord[i])  # E

            if hc[0, i] > cyclic_length / 2.0:
                hc[0, i] = hc[0, i] - cyclic_length
            if hc[0, i] < -cyclic_length / 2.0:
                hc[0, i] = hc[0, i] + cyclic_length

            if hc[2, i] > cyclic_length / 2.0:
                hc[2, i] = hc[2, i] - cyclic_length
            if hc[2, i] < -cyclic_length / 2.0:
                hc[2, i] = hc[2, i] + cyclic_length

            hc[0, i] *= Mt[i] * r_earth
            hc[2, i] *= Mt[i] * r_earth

        hh[1, :] = (hc[1, :] + hc[3, :]) / 2
        hh[3, :] = (hc[1, :] + hc[3, :]) / 2
        hh[0, :] = (hc[0, :] + hc[2, :]) / 2
        hh[2, :] = (hc[0, :] + hc[2, :]) / 2

        area = hh[3, :] * hh[2, :]

        ss = np.zeros(nza, dtype="float")
        ii = np.zeros(nza, dtype="int")
        jj = np.zeros(nza, dtype="int")

        nn = 0
        for n in range(e2d):
            no = nn
            for m in range(4):
                if ee_pos[m, n] != n and self._mask_n[ee_pos[m, n]] != 0:
                    nn += 1
                    # print(f"nn: {nn} m: {m} n: {n}")
                    ss[nn] = (hc[m, n] / hh[m, n]) / area[n]
                    ii[nn] = n
                    jj[nn] = ee_pos[m, n]

            ii[no] = n
            jj[no] = n
            ss[no] = -np.sum(ss[no : nn + 1])
            nn += 1

        self._ss = ss
        self._ii = ii
        self._jj = jj
        self._area = area

        # Create a mask where both _ii and _jj are not 0
        mask_sp = np.logical_and(self._mask_n[ii], self._mask_n[jj])

        self._ss = self._ss[mask_sp]
        self._ii = self._ii[mask_sp]
        self._jj = self._jj[mask_sp]

        self.set_backend("gpu" if gpu else "cpu")

    def get_backend(self) -> str:
        """
        Get current computational backend.

        Returns
        -------
        str
            Current backend ('cpu' or 'gpu').
        """
        return self._backend

    def set_backend(self, backend: str):
        """
        Set computational backend for filtering operations.

        Parameters
        ----------
        backend : str
            Desired backend ('cpu' or 'gpu').

        Notes
        -----
        Configures appropriate sparse linear algebra functions for the backend.
        """
        self.csc_matrix, self.identity, self.cg, self.convers, self.tonumpy = (
            get_backend(backend)
        )
        self._backend = backend

    def _compute(
        self,
        n: int,
        k: float,
        data: np.ndarray,
        maxiter: int = 150_000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        Smat1 = self.csc_matrix(
            (
                self.convers(self._ss) * (-1.0 / np.square(k)),
                (self.convers(self._ii), self.convers(self._jj)),
            ),
            shape=(self._e2d, self._e2d),
        )
        Smat = self.identity(self._e2d) + 2.0 * (Smat1**n)

        ttu = self.convers(data)
        ttw = ttu - Smat @ ttu  # Work with perturbations

        b = 1.0 / Smat.diagonal()  # Simple preconditioner
        arr = self.convers(np.arange(self._e2d))
        pre = self.csc_matrix((b, (arr, arr)), shape=(self._e2d, self._e2d))

        tts, code = self.cg(Smat, ttw, None, tol, maxiter, pre)
        if code != 0:
            raise SolverNotConvergedError(
                "Solver has not converged without metric terms",
                [f"output code with code: {code}"],
            )

        tts += ttu
        return self.tonumpy(tts)

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to scalar field on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float
            Filter wavelength in spatial units.
        data : np.ndarray
            Scalar field values on grid (shape: (nx, ny)).

        Returns
        -------
        np.ndarray
            Filtered scalar field (shape: (nx, ny)).

        Raises
        ------
        ValueError
            If filter order n < 1.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")

        return np.reshape(
            self._compute(n, k, np.reshape(data, self._e2d)), (self._nx, self._ny)
        )

    def compute_velocity(
        self, n: int, k: float, ux: np.ndarray, vy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to velocity components on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float
            Filter wavelength in spatial units.
        ux : np.ndarray
            Eastward velocity component (shape: (nx, ny)).
        vy : np.ndarray
            Northward velocity component (shape: (nx, ny)).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered velocity components (ux_filt, vy_filt) each with shape (nx, ny).

        Raises
        ------
        ValueError
            If filter order n < 1.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")

        return (
            np.reshape(
                self._compute(n, k, np.reshape(ux, self._e2d)), (self._nx, self._ny)
            ),
            np.reshape(
                self._compute(n, k, np.reshape(vy, self._e2d)), (self._nx, self._ny)
            ),
        )

    def compute_spectra_scalar(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute power spectra for scalar field at specified wavelengths.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        data : np.ndarray
            Scalar field values on grid (shape: (nx, ny)).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.

        Returns
        -------
        np.ndarray
            Power spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total variance
            [1:] : Variance at each wavelength k
        """
        nr = len(k)
        tt = np.reshape(data, self._e2d)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask: np.ndarray = np.zeros(tt.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]

        spectra[0] = np.sum(selected_area * (np.square(tt))[not_mask]) / np.sum(
            selected_area
        )

        for i in range(nr):
            ttu = self._compute(n, k[i], tt)
            ttu -= tt

            ttu[mask] = 0.0
            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu))[not_mask]
            ) / np.sum(selected_area)

        return spectra

    def compute_spectra_velocity(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute power spectra for velocity field at specified wavelengths.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        ux : np.ndarray
            Eastward velocity component (shape: (nx, ny)).
        vy : np.ndarray
            Northward velocity component (shape: (nx, ny)).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.

        Returns
        -------
        np.ndarray
            Kinetic energy spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total kinetic energy
            [1:] : Kinetic energy at each wavelength k
        """

        nr = len(k)
        unod = np.reshape(ux, self._e2d)
        vnod = np.reshape(vy, self._e2d)

        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(unod.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]
        spectra[0] = np.sum(
            selected_area * (np.square(unod) + np.square(vnod))[not_mask]
        ) / np.sum(selected_area)

        for i in range(nr):
            ttu = self._compute(n, k[i], unod)
            ttv = self._compute(n, k[i], vnod)

            ttu -= unod
            ttv -= vnod

            ttu[mask] = 0.0
            ttv[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu) + np.square(ttv))[not_mask]
            ) / np.sum(selected_area)

        return spectra
