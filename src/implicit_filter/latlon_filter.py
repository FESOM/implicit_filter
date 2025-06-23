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
    A filter class for data based on regular latitude and longitude grids using NumPy arrays.

    Methods
    -------
    many_compute(n: int, k: float, data: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        Placeholder method to compute filtering on multiple datasets. Not implemented yet.

    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Computes the filtered velocity fields.

    compute(n: int, k: float, data: np.ndarray) → np.ndarray:
        Computes the filtered data.

    """

    def __init__(self, *initial_data, **kwargs):
        """
        Initializes the LatLonNumpyFilter with the given data and keyword arguments.

        Parameters
        ----------
        initial_data : tuple
            Initial data to be passed to the parent class.
        kwargs : dict
            Additional keyword arguments.

        """
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
        mask: np.ndarray | None = None,
        gpu: bool = False,
    ):
        """
        Prepares the filter for latitude and longitude grids regular grids

        Parameters
        ----------
        latitude: np.ndarray
            1D np.ndarray of floats with latitude values
        longitude: np.ndarray
            1D np.ndarray of floats with longitude values
        cartesian: bool
            If true, the conversion from degrees to km should assume that mesh is cartesian.
        local: bool
            If true, neighborhood calculation doesn't wrap around an East/West direction
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

        self._mask_n = np.ones(self._e2d, dtype=bool) if mask is None else mask

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
        cyclic_length = (
            360  # in degrees; if not cyclic, take it larger than  zonal size
        )
        cyclic_length = cyclic_length * math.pi / 180
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
                if ee_pos[m, n] != n:
                    nn += 1
                    # print(f"nn: {nn} m: {m} n: {n}")
                    ss[nn] = (hc[m, n] / hh[m, n]) / area[n]
                    ii[nn] = n
                    jj[nn] = ee_pos[m, n]

            ii[no] = n
            jj[no] = n
            ss[no] = -np.sum(ss[no : nn + 1])
            nn += 1

        # Create a mask where both _ii and _jj are not 0
        mask_sp = self._mask_n[self._ii] & self._mask_n[self._jj]

        self._ss = self._ss[mask_sp]
        self._ii = self._ii[mask_sp]
        self._jj = self._jj[mask_sp]

        self._ss = ss
        self._ii = ii
        self._jj = jj
        self._area = area

        self.set_backend("gpu" if gpu else "cpu")

    def get_backend(self) -> str:
        return self._backend

    def set_backend(self, backend: str):
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
        if n < 1:
            raise ValueError("Filter order must be positive")

        return np.reshape(
            self._compute(n, k, np.reshape(data, self._e2d)), (self._nx, self._ny)
        )

    def compute_velocity(
        self, n: int, k: float, ux: np.ndarray, vy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            ttu, ttv = self.compute_velocity(n, k[i], unod, vyn)

            ttu -= unod
            ttv -= vnod

            ttu[mask] = 0.0
            ttv[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu) + np.square(ttv))[not_mask]
            ) / np.sum(selected_area)

        return spectra
