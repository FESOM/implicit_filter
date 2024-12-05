import math
from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg

from implicit_filter._numpy_functions import calculate_local_regular_neighbourhood, \
    calculate_global_regular_neighbourhood
from implicit_filter.filter import Filter
from implicit_filter._utils import SolverNotConvergedError, VeryStupidIdeaError, transform_attribute


class LatLonNumpyFilter(Filter):
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

        # Transform and initialize attributes with default values
        transform_attribute(self, "_e2d", it, 0)
        transform_attribute(self, "_nx", it, 0)
        transform_attribute(self, "_ny", it, 0)
        transform_attribute(self, "_ss", ar, None)
        transform_attribute(self, "_ii", ar, None)
        transform_attribute(self, "_jj", ar, None)

    def prepare(self, latitude: np.ndarray, longitude: np.ndarray, cartesian: bool = False, local: bool = True):
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

        if local:
            ee_pos, nza = calculate_local_regular_neighbourhood(e2d, nx, ny)
        else:
            ee_pos, nza = calculate_global_regular_neighbourhood(e2d, nx, ny)

        rad = math.pi / 180.0

        if cartesian:
            Mt = np.ones(e2d)
        else:
            Mt = np.cos(rad * ycoord[ee_pos])

        hh = np.ones((4, e2d))  # Edge lengths
        hc = np.ones((4, e2d))  # Distance to next cell centers
        r_earth = 6400.
        cyclic_length = 360  # in degrees; if not cyclic, take it larger than  zonal size
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
            ss[no] = -np.sum(ss[no:nn + 1])
            nn += 1

        self._ss = ss
        self._ii = ii
        self._jj = jj

    def _compute(self, n, k, data: np.ndarray, maxiter=150_000, tol=1e-6) -> np.ndarray:
        e2d = self._e2d

        Smat1 = csc_matrix((self._ss * (1.0 / k ** 2), (self._ii, self._jj)), shape=(e2d, e2d))
        Smat2 = identity(e2d)
        Smat = Smat2 + 0.5 * (-1 * Smat1) ** n
        ttw = data.T - Smat @ data.T  # Work with perturbations

        b = 1. / Smat.diagonal()  # Simple preconditioner
        pre = csc_matrix((b, (np.arange(e2d), np.arange(e2d))), shape=(e2d, e2d))
        tts, code = cg(Smat, ttw, maxiter=maxiter, rtol=tol, M=pre)
        tts += data.T

        if code != 0:
            raise SolverNotConvergedError("Solver has not converged",
                                          [f"output code with code: {code}"])
        return tts

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        return np.reshape(self._compute(n, k, np.reshape(data, self._e2d)), (self._nx, self._ny))

    def compute_velocity(self, n: int, k: float, ux: np.ndarray, uy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        return (np.reshape(self._compute(n, k, np.reshape(ux, self._e2d)), (self._nx, self._ny)),
                np.reshape(self._compute(n, k, np.reshape(vy, self._e2d)), (self._nx, self._ny)))

    def many_compute(self, n: int, k: float, data: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        raise NotImplementedError("This method is not yet implemented.")
