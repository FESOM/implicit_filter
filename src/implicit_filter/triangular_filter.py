from math import pi
from typing import Tuple, Iterable

import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax

jax.config.update("jax_platforms", "cpu")

from implicit_filter.utils._auxiliary import (
    neighboring_triangles,
    neighbouring_nodes,
    areas,
)
from implicit_filter.utils._jax_function import (
    make_smooth,
    make_smat,
    make_smat_full,
    transform_mask_to_nodes,
)
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    transform_attribute,
    get_backend,
)
from implicit_filter.filter import Filter


class TriangularFilter(Filter):
    """
    A class for filtering data using JAX for generic triangular meshes.
    Extends the base Filter class.

    Attributes:
    -----------
    _elem_area : Optional[jnp.ndarray]
        Area of each element in the mesh.
    _area : Optional[jnp.ndarray]
        Area of each node's neighborhood in the mesh.
    _ne_pos : Optional[jnp.ndarray]
        Connectivity matrix representing neighboring elements for each node.
    _ne_num : Optional[jnp.ndarray]
        Number of neighboring elements for each node.
    _dx : Optional[jnp.ndarray]
        X-component of the derivative of P1 basis functions.
    _dy : Optional[jnp.ndarray]
        Y-component of the derivative of P1 basis functions.
    _ss : Optional[jnp.ndarray]
        Non-zero entries of the sparse matrix.
    _ii : Optional[jnp.ndarray]
        Row indices of non-zero entries.
    _jj : Optional[jnp.ndarray]
        Column indices of non-zero entries.
    _n2d : int
        Total number of nodes in the mesh.
    _full : bool
        Flag indicating whether to use the full matrix.
    _mask_n: Optional[jnp.ndarray]
        Mask of valid elements in the mesh.
        For example it can land mask.


    Methods:
    --------
    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute filtered velocity components (u, v) using implicit filtering.
    Compute(n: int, k: float, data: np.ndarray) -> np.ndarray:
        Compute filtered data using implicit filtering.
    Prepare(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
            cartesian: bool, cyclic_length: float, full: bool = False):
        Prepare the filter for a specific mesh.

    """

    def __init__(self, *initial_data, **kwargs):
        """
        Initialize the Triangular filter instance.

        Parameters:
        -----------
        *initial_data : positional arguments
            Positional arguments passed to the base class constructor.
        **kwargs : keyword arguments
            Keyword arguments passed to the base class constructor.
        """
        self.backend = "cpu"  # If initial_data doesn't contain backend, cpu is used
        super().__init__(initial_data, kwargs)
        # Transform to JAX array
        jx = lambda ar: jnp.array(ar)
        bl = lambda ar: bool(ar)
        it = lambda ar: int(ar)
        st = lambda ar: str(ar)

        transform_attribute(self, "_elem_area", jx, None)
        transform_attribute(self, "_area", jx, None)
        transform_attribute(self, "_ne_pos", jx, None)
        transform_attribute(self, "_en_pos", jx, None)
        transform_attribute(self, "_ne_num", jx, None)
        transform_attribute(self, "_dx", jx, None)
        transform_attribute(self, "_dy", jx, None)

        transform_attribute(self, "_ss", jx, None)
        transform_attribute(self, "_ii", jx, None)
        transform_attribute(self, "_jj", jx, None)
        transform_attribute(self, "_mask_n", jx, None)

        transform_attribute(self, "_n2d", it, 0)
        transform_attribute(self, "_e2d", it, 0)
        transform_attribute(self, "_full", bl, False)
        transform_attribute(self, "_backend", st, "cpu")

        self.set_backend(self.backend)

    def _compute(self, n, kl, ttu, tol=1e-6, maxiter=150000) -> np.ndarray:
        Smat1 = self.csc_matrix(
            (
                self.convers(self._ss) * (1.0 / np.square(kl)),
                (self.convers(self._ii), self.convers(self._jj)),
            ),
            shape=(self._n2d, self._n2d),
        )
        Smat = self.identity(self._n2d) + 2.0 * (Smat1**n)

        ttu = self.convers(ttu)
        ttw = ttu - Smat @ ttu  # Work with perturbations

        b = 1.0 / Smat.diagonal()  # Simple preconditioner
        arr = self.convers(np.arange(self._n2d))
        pre = self.csc_matrix((b, (arr, arr)), shape=(self._n2d, self._n2d))

        tts, code = self.cg(Smat, ttw, None, tol, maxiter, pre)
        if code != 0:
            raise SolverNotConvergedError(
                "Solver has not converged without metric terms",
                [f"output code with code: {code}"],
            )

        tts += ttu
        return self.tonumpy(tts)

    def _compute_full(self, n, kl, ttuv, tol=1e-5, maxiter=150000) -> np.ndarray:
        Smat1 = self.csc_matrix(
            (
                self.convers(self._ss) * (1.0 / np.square(kl)),
                (self.convers(self._ii), self.convers(self._jj)),
            ),
            shape=(2 * self._n2d, 2 * self._n2d),
        )
        Smat = self.identity(2 * self._n2d) + 2.0 * (Smat1**n)

        ttuv = self.convers(ttuv)
        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        b = 1.0 / Smat.diagonal()  # Simple preconditioner
        arr = self.convers(np.arange(2 * self._n2d))
        pre = self.csc_matrix((b, (arr, arr)), shape=(2 * self._n2d, 2 * self._n2d))

        tts, code = self.cg(Smat, ttw, None, tol, maxiter, pre)
        if code != 0:
            raise SolverNotConvergedError(
                "Solver has not converged with metric terms",
                [f"output code with code: {code}"],
            )

        tts += ttuv
        return self.tonumpy(tts)

    def compute_velocity(
        self, n: int, k: float, ux: np.ndarray, vy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to velocity components on triangular mesh.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float
            Filter wavelength in spatial units.
        ux : np.ndarray
            Eastward velocity component at mesh nodes.
        vy : np.ndarray
            Northward velocity component at mesh nodes.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered velocity components (ux_filt, vy_filt).

        Raises
        ------
        ValueError
            If filter order n < 1.
        SolverNotConvergedError
            If linear solver fails to converge.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")

        uxn = ux
        vyn = vy

        if self._full:
            ttuv = self._compute_full(n, k, np.concatenate((uxn, vyn)))
            return ttuv[0 : self._n2d], ttuv[self._n2d : 2 * self._n2d]
        else:
            ttu = self._compute(n, k, uxn)
            ttv = self._compute(n, k, vyn)
            return ttu, ttv

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to scalar field on triangular mesh.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float
            Filter wavelength in spatial units.
        data : np.ndarray
            Scalar field values at mesh nodes.

        Returns
        -------
        np.ndarray
            Filtered scalar field.

        Raises
        ------
        ValueError
            If filter order n < 1.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")

        return np.array(
            self._compute_full(n, k, data) if self._full else self._compute(n, k, data)
        )

    def prepare(
        self,
        n2d: int,
        e2d: int,
        tri: np.ndarray,
        xcoord: np.ndarray,
        ycoord: np.ndarray,
        meshtype: str = "m",
        cartesian: bool = True,
        cyclic_length: float = 360.0 * pi / 180.0,
        full: bool = False,
        mask: np.ndarray = None,
        gpu: bool = False,
    ):
        """
        Prepare filter for a specific triangular mesh.

        Computes mesh topology, geometric properties, and assembles the filter
        operator matrix. Must be called before any filtering operations.

        Parameters
        ----------
        n2d : int
            Number of nodes in the mesh.
        e2d : int
            Number of elements in the mesh.
        tri : np.ndarray
            Element connectivity matrix (e2d x 3) of node indices.
        xcoord : np.ndarray
            X-coordinates of mesh nodes (degrees).
        ycoord : np.ndarray
            Y-coordinates of mesh nodes (degrees).
        meshtype : str, optional
            Mesh type identifier ('m' for general meshes).
        cartesian : bool, optional
            True for Cartesian coordinates, False for spherical.
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        full : bool, optional
            True to include metric terms in operator (default: False).
        mask : np.ndarray, optional
            Element mask where True indicates land (default: all ocean).
        gpu : bool, optional
            True to enable GPU acceleration (default: False).

        Notes
        -----
        Coordinates are expected in degrees while cyclic_length is in radians.
        The mask is converted to nodal representation where True indicates land.
        """
        # NOTE: xcoord & ycoord are in degrees, but cyclic_length is in radians
        self._n2d = n2d
        self._e2d = e2d
        self._full = full

        if mask is None:
            mask = np.ones(e2d)
        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(
            n2d,
            e2d,
            tri,
            xcoord,
            ycoord,
            ne_num,
            ne_pos,
            meshtype,
            cartesian,
            cyclic_length,
            mask,
        )

        self._elem_area = jnp.array(elem_area)
        self._dx = jnp.array(dx)
        self._dy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        self._en_pos = jnp.array(tri.T)  # element positions in terms of nodes
        self._ne_num = jnp.array(ne_num)
        self._ne_pos = jnp.array(ne_pos)
        self._area = jnp.array(area)

        self._mask_n = jnp.array(mask)

        self._mask_n = transform_mask_to_nodes(
            self._mask_n, self._ne_pos, self._ne_num, self._n2d
        )
        self._mask_n = ~jnp.where(self._mask_n > 0.5, 1.0, 0.0).astype(
            bool
        )  # Where there's ocean

        smooth, metric = make_smooth(
            jMt,
            self._elem_area,
            self._dx,
            self._dy,
            jnn_num,
            jnn_pos,
            jtri,
            n2d,
            e2d,
            full,
        )

        smooth = vmap(lambda n: smooth[:, n] / self._area[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / self._area[n])(jnp.arange(0, n2d)).T

        self._ss, self._ii, self._jj = (
            make_smat_full(jnn_pos, jnn_num, smooth, metric, n2d, int(jnp.sum(jnn_num)))
            if full
            else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        )

        ## Set rows (and columns!) of smooth where (node) mask is 0 (land) to 0: This enforces a Neumann BC
        #   i.e. Set _ss = 0 where mask_n[_ii] = 0 && mask_n[_jj] = 0
        # AFW
        # Create a mask where both _ii and _jj are not 0
        if full:
            mask_sp = jnp.logical_and(
                self._mask_n[self._ii % n2d], self._mask_n[self._jj % n2d]
            )
        else:
            mask_sp = jnp.logical_and(self._mask_n[self._ii], self._mask_n[self._jj])

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
        return self.backend

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
        self.backend = backend

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
            Scalar field values at mesh nodes.
        mask : np.ndarray, optional
            Boolean mask where True excludes nodes from spectra computation.

        Returns
        -------
        np.ndarray
            Power spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total variance
            [1:] : Variance at each wavelength k
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask: np.ndarray = np.zeros(data.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]

        spectra[0] = np.sum(selected_area * (np.square(data))[not_mask]) / np.sum(
            selected_area
        )

        for i in range(nr):
            ttu = self.compute(n, k[i], data)
            ttu -= data

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
            Eastward velocity component at mesh nodes.
        vy : np.ndarray
            Northward velocity component at mesh nodes.
        mask : np.ndarray, optional
            Boolean mask where True excludes nodes from spectra computation.

        Returns
        -------
        np.ndarray
            Kinetic energy spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total kinetic energy
            [1:] : Kinetic energy at each wavelength k

        Notes
        -----
        Implements spectral decomposition:
            E(k) = ⟨(u - uₖ)² + (v - vₖ)²⟩
        where (uₖ, vₖ) is the filtered velocity at wavelength k.
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(ux.shape, dtype=bool)

        unod = ux
        vnod = vy

        not_mask = ~mask
        selected_area = self._area[not_mask]
        spectra[0] = np.sum(
            selected_area * (np.square(unod) + np.square(vnod))[not_mask]
        ) / np.sum(selected_area)

        for i in range(nr):
            ttu, ttv = self.compute_velocity(n, k[i], unod, vnod)

            ttu -= unod
            ttv -= vnod

            ttu[mask] = 0.0
            ttv[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu) + np.square(ttv))[not_mask]
            ) / np.sum(selected_area)

        return spectra
