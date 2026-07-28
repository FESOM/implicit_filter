import math
from typing import Tuple, Iterable

import numpy as np

from implicit_filter.utils._auxiliary import R_EARTH
from implicit_filter.utils._numpy_functions import (
    calculate_local_regular_neighbourhood,
    calculate_global_regular_neighbourhood,
)
from implicit_filter.filter import Filter
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    transform_attribute,
)
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


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

    _mask_n : np.ndarray
        Boolean mask for valid grid points (False indicates land)
    """

    def __init__(self, *initial_data, **kwargs):
        super().__init__(*initial_data, **kwargs)
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
        r_earth = R_EARTH

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



    def _compute(
        self,
        n: int,
        k: float | np.ndarray,
        data: np.ndarray,
        x0: np.ndarray | None = None,
        maxiter: int = 150_000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        k_arg = k  # pre-broadcast value; the V-cycle path needs a scalar k
        if isinstance(k, (float, int, np.number)):
            k = np.ones(self._e2d) * k

        scaling_vector = -1.0 / np.square(k)
        data_smat1 = self._ss * scaling_vector[self._jj]

        def apply_A(x):
            y = x
            for _ in range(n):
                y = jnp.zeros_like(x).at[self._ii].add(data_smat1 * y[self._jj])
            return x + 2.0 * y

        diag_mask = self._ii == self._jj
        Smat1_diag = jnp.zeros(self._e2d).at[self._ii[diag_mask]].add(data_smat1[diag_mask])
        approx_diag_Smat = 1.0 + 2.0 * (Smat1_diag ** n)
        
        def precond(x):
            return x / approx_diag_Smat

        ttu = jnp.array(data)
        ttw = ttu - apply_A(ttu)  # Work with perturbations

        x0_pert = None if x0 is None else (jnp.array(x0) - ttu)

        if self.get_preconditioner() == "vcycle":
            from implicit_filter.utils._vcycle import (
                solve_with_vcycle, validate_scalar_k)

            # The lat-lon stencil is assembled negative-semidefinite (the
            # solve scales by -1/k^2), so the PSD-convention stencil is -S.
            tts = solve_with_vcycle(
                ss=-np.asarray(self._ss), ii=self._ii, jj=self._jj,
                area=self._area, n_size=int(self._e2d), n=n,
                k=validate_scalar_k(k_arg), apply_A=apply_A,
                b_pert=ttw, x0_pert=x0_pert, tol=tol, maxiter=maxiter,
                options=self.preconditioner_options,
                cache=self.vcycle_cache, tag="latlon")
        else:
            M = precond if self.get_preconditioner() == "jacobi" else None
            tts, code = cg(apply_A, ttw, x0=x0_pert, tol=tol, maxiter=maxiter, M=M)
            if code is not None and code != 0:
                raise SolverNotConvergedError(
                    "Solver has not converged without metric terms",
                    [f"output code with code: {code}"],
                )

        tts += ttu
        return np.array(tts)

    def compute(self, n: int, k: float | np.ndarray, data: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """
        Apply filter to scalar field on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data
        data : np.ndarray
            Scalar field values on grid (shape: (nx, ny)).
        x0 : np.ndarray | None
            Initial guess for the solver.

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

        x0_flat = np.reshape(x0, self._e2d) if x0 is not None else None
        return np.reshape(
            self._compute(n, k, np.reshape(data, self._e2d), x0=x0_flat), (self._nx, self._ny)
        )

    def compute_velocity(
        self, n: int, k: float | np.ndarray, ux: np.ndarray, vy: np.ndarray, ux0: np.ndarray | None = None, vy0: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to velocity components on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data
        ux : np.ndarray
            Eastward velocity component (shape: (nx, ny)).
        vy : np.ndarray
            Northward velocity component (shape: (nx, ny)).
        ux0 : np.ndarray | None
            Initial guess for ux.
        vy0 : np.ndarray | None
            Initial guess for vy.

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

        ux0_flat = np.reshape(ux0, self._e2d) if ux0 is not None else None
        vy0_flat = np.reshape(vy0, self._e2d) if vy0 is not None else None

        return (
            np.reshape(
                self._compute(n, k, np.reshape(ux, self._e2d), x0=ux0_flat), (self._nx, self._ny)
            ),
            np.reshape(
                self._compute(n, k, np.reshape(vy, self._e2d), x0=vy0_flat), (self._nx, self._ny)
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

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

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

        x0 = None
        for i in range(nr):
            ttu = self._compute(n, k[i], tt, x0=x0)
            x0 = ttu
            
            ttu_diff = ttu - tt
            ttu_diff[mask] = 0.0
            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu_diff))[not_mask]
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

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

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

        ux0, vy0 = None, None
        for i in range(nr):
            ttu = self._compute(n, k[i], unod, x0=ux0)
            ttv = self._compute(n, k[i], vnod, x0=vy0)
            ux0, vy0 = ttu, ttv

            ttu_diff = ttu - unod
            ttv_diff = ttv - vnod

            ttu_diff[mask] = 0.0
            ttv_diff[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu_diff) + np.square(ttv_diff))[not_mask]
            ) / np.sum(selected_area)

        return spectra
