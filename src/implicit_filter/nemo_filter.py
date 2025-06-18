from typing import Tuple, List

import numpy as np
import xarray as xr
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg

from implicit_filter.utils._auxiliary import find_adjacent_points_north
from implicit_filter.utils._numpy_functions import calculate_global_nemo_neighbourhood
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    get_backend,
    transform_attribute,
)
from .filter import Filter


class NemoFilter(Filter):
    """
    A filter class for NEMO ocean model data using NumPy arrays.

    Methods
    -------
    many_compute(n: int, k: float, data: np.ndarray | List[np.ndarray]) -> List[np.ndarray]:
        Placeholder method to compute filtering on multiple datasets. Not implemented yet.

    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Computes the filtered velocity fields.

    compute(n: int, k: float, data: np.ndarray) -> np.ndarray:
        Computes the filtered data.

    prepare_from_file(file: str, vl: int):
        Prepares the filter using data from a specified file.

    """

    def __init__(self, *initial_data, **kwargs):
        """
        Initializes the NemoNumpyFilter with the given data and keyword arguments.

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
        st = lambda st: str(st)

        # Transform and initialize attributes with default values
        transform_attribute(self, "_e2d", it, 0)
        transform_attribute(self, "_nx", it, 0)
        transform_attribute(self, "_ny", it, 0)
        transform_attribute(self, "_area", ar, None)
        transform_attribute(self, "_backend", st, "cpu")

        self.set_backend(self._backend)

    def compute_velocity(
        self, n: int, k: float, ux: np.ndarray, vy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")

        v = np.zeros((4, self._e2d))  # N, W, E, S
        v[0, :] = np.reshape(ux, self._e2d)  # West
        v[1, :] = np.reshape(vy, self._e2d)  # North

        for i in range(self._e2d):
            if self._ee_pos[3, i] != i:
                v[3, i] = v[1, self._ee_pos[3, i]]

            if self._ee_pos[2, i] != i:
                v[2, i] = v[0, self._ee_pos[2, i]]

        ttu = v[0, :] + v[2, :]
        ttv = v[1, :] + v[3, :]

        return (
            np.reshape(self._compute(n, k, ttu), (self._nx, self._ny)),
            np.reshape(self._compute(n, k, ttv), (self._nx, self._ny)),
        )

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        if n < 1:
            raise ValueError("Filter order must be positive")

        tt = np.reshape(data, self._e2d)
        return np.reshape(self._compute(n, k, tt), (self._nx, self._ny))

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

    def prepare_from_file(self, file: str, vl: int, gpu: bool = False):
        ds = xr.open_dataset(file)

        nx, ny = (
            ds.gphit.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values.shape
        )
        north_adj, _ = find_adjacent_points_north(file, 1e-5)
        e2d = nx * ny

        self._nx = nx
        self._ny = ny
        self._e2d = e2d

        ee_pos, nza = calculate_global_nemo_neighbourhood(e2d, nx, ny, north_adj)
        self._ee_pos = ee_pos

        # Cell sizes
        hx = np.reshape(
            ds.e1t.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        hy = np.reshape(
            ds.e2t.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        self._area = hx * hy

        hh = np.ones((4, e2d))  # Edge lengths
        hh[1, :] = np.reshape(
            ds.e2u.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # North edge
        hh[0, :] = np.reshape(
            ds.e1v.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # West edge
        for n in range(e2d):
            if ee_pos[3, n] != n:
                hh[3, n] = hh[1, ee_pos[3, n]]
            else:
                hh[3, n] = hh[1, n]

            if ee_pos[2, n] != n:
                hh[2, n] = hh[0, ee_pos[2, n]]
            else:
                hh[2, n] = hh[0, n]

        # Cell heights
        h3u = np.reshape(
            ds.e3u_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3v = np.reshape(
            ds.e3v_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3t = np.reshape(
            ds.e3t_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )

        hc = np.ones((4, e2d))  # Distance to next cell centers
        hc[0, :] = np.reshape(
            ds.e1u.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # West neighbour
        hc[1, :] = np.reshape(
            ds.e2v.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # North neighbour

        for n in range(e2d):
            if ee_pos[3, n] != n:
                hc[3, n] = hc[1, ee_pos[1, n]]
            else:
                hc[3, n] = hc[1, n]

            if ee_pos[2, n] != n:
                hc[2, n] = hh[2, ee_pos[2, n]]
            else:
                hc[2, n] = hc[2, n]

        ss = np.zeros(nza, dtype="float")
        ii = np.zeros(nza, dtype="int")
        jj = np.zeros(nza, dtype="int")
        mask = np.reshape(
            ds.tmask.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values,
            nx * ny,
        )

        nn = 0
        for n in range(e2d):
            no = nn
            for m in range(4):
                if ee_pos[m, n] != n and mask[ee_pos[m, n]] != 0:
                    nn += 1
                    ss[nn] = (
                        (hh[m, n] * h3u[n]) / (hc[m, n] * h3t[n])
                        if m % 2 == 0
                        else (hh[m, n] * h3v[n]) / (hc[m, n] * h3t[n])
                    )
                    ss[nn] /= self._area[n]  # Add division on cell area if you prefer
                    ii[nn] = n
                    jj[nn] = ee_pos[m, n]

            ii[no] = n
            jj[no] = n
            ss[no] = -np.sum(ss[no : nn + 1])
            nn += 1

        self._ss = ss
        self._ii = ii
        self._jj = jj

        self.set_backend("gpu" if gpu else "cpu")

    def get_backend(self) -> str:
        return self._backend

    def set_backend(self, backend: str):
        self.csc_matrix, self.identity, self.cg, self.convers, self.tonumpy = (
            get_backend(backend)
        )
        self._backend = backend
