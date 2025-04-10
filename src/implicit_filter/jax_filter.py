import concurrent
import math
from typing import Tuple, List
import numpy as np
import jax.numpy as jnp
import xarray as xr
from jax import vmap
from ._auxiliary import neighboring_triangles, neighbouring_nodes, areas
from ._jax_function import make_smooth, make_smat, make_smat_full, transform_veloctiy_to_nodes
from ._utils import VeryStupidIdeaError, SolverNotConvergedError, TheHollyHandErrorOfAntioch, transform_attribute
from implicit_filter.filter import Filter
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg


class JaxFilter(Filter):
    """
    A class for filtering data using JAX for accelerating implicit filtering techniques.
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

    Methods:
    --------
    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute filtered velocity components (u, v) using implicit filtering.
    compute(n: int, k: float, data: np.ndarray) -> np.ndarray:
        Compute filtered data using implicit filtering.
    prepare(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
            carthesian: bool, cyclic_length: float, full: bool = False):
        Prepare the filter for a specific mesh.

    """

    def __init__(self, *initial_data, **kwargs):
        """
        Initialize the JaxFilter instance.

        Parameters:
        -----------
        *initial_data : positional arguments
            Positional arguments passed to the base class constructor.
        **kwargs : keyword arguments
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(initial_data, kwargs)
        # Transform to JAX array
        jx = lambda ar: jnp.array(ar)
        bl = lambda ar: bool(ar)
        it = lambda ar: int(ar)

        transform_attribute(self, "_elem_area", jx, None)
        transform_attribute(self, "_area", jx, None)
        transform_attribute(self, "_ne_pos", jx, None)
        transform_attribute(self, "_ne_num", jx, None)
        transform_attribute(self, "_dx", jx, None)
        transform_attribute(self, "_dy", jx, None)

        transform_attribute(self, "_ss", jx, None)
        transform_attribute(self, "_ii", jx, None)
        transform_attribute(self, "_jj", jx, None)

        transform_attribute(self, "_n2d", it, 0)
        transform_attribute(self, "_full", bl, False)

    def _compute(self, n, kl, ttu, tol=1e-6, maxiter=150000) -> np.ndarray:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)

        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged without metric terms",
                                          [f"output code with code: {code}"])

        tts += ttu
        return np.array(tts)

    def _many_compute(self, n, kl, data, tol=1e-6, maxiter=150000) -> List[np.ndarray]:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)
        output: List[np.ndarray] = list()
        for ttu in data:
            ttw = ttu - Smat @ ttu  # Work with perturbations
            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)

            if code != 0:
                raise SolverNotConvergedError("Solver has not converged without metric terms",
                                              [f"output code with code: {code}"])

            tts += ttu
            output.append(tts)

        return output

    def _compute_full(self, n, kl, ttuv, tol=1e-6, maxiter=150000) -> np.ndarray:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = identity(2 * self._n2d) + 0.5 * (Smat1 ** n)

        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged with metric terms",
                                          [f"output code with code: {code}"])

        tts += ttuv
        return np.array(tts)

    def _many_compute_full(self, n, kl, ux, vy, tol=1e-6, maxiter=150000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = identity(2 * self._n2d) + 0.5 * (Smat1 ** n)

        oux: List[np.ndarray] = list()
        ovy: List[np.ndarray] = list()

        for i in range(len(ux)):
            ttuv = jnp.concatenate((ux[i], vy[i]))
            ttw = ttuv - Smat @ ttuv  # Work with perturbations

            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
            if code != 0:
                raise SolverNotConvergedError("Solver has not converged with metric terms",
                                              [f"output code with code: {code}"])

            tts += ttuv

            oux.append(tts[0:self._n2d])
            ovy.append(tts[self._n2d:2 * self._n2d])

        return oux, ovy

    def compute_velocity(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n == 3:
            raise TheHollyHandErrorOfAntioch()
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn, vyn = transform_veloctiy_to_nodes(jnp.array(ux), jnp.array(vy), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)
        if self._full:
            ttuv = self._compute_full(n, k, jnp.concatenate((uxn, vyn)))
            return ttuv[0:self._n2d], ttuv[self._n2d:2*self._n2d]
        else:
            ttu = self._compute(n, k, uxn)
            ttv = self._compute(n, k, vyn)
            return ttu, ttv

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        return np.array(self._compute_full(n, k, data) if self._full else self._compute(n, k, data))

    def many_compute(self, n: int, k: float, data: np.ndarray | List[np.ndarray]) -> List[np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        if type(data) is np.ndarray:
            if len(data.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            return self._many_compute(n, k, data.T)
        elif type(data) is list:
            return self._many_compute(n, k, data)
        else:
            raise ValueError("Input data is of incorrect type")


    def many_compute_velocity(self, n: int, k: float, ux: np.ndarray | List[np.ndarray],
                              vy: np.ndarray | List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn = []
        vyn = []
        futures = []

        if type(ux) is np.ndarray:
            if len(ux.shape) != 2 and len(vy.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux.T)):
                    futures.append(executor.submit(transform_veloctiy_to_nodes, jnp.array(ux[:, i]), jnp.array(vy[:, i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)

        elif type(ux) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux)):
                    futures.append(executor.submit(transform_veloctiy_to_nodes, jnp.array(ux[i]), jnp.array(vy[i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")

        for f in futures:
            tmp_u, tmp_v = f.result()
            uxn.append(tmp_u)
            vyn.append(tmp_v)

        if self._full:
            ttu, ttv = self._many_compute_full(n, k, uxn, vyn)
        else:
            ttu = self._many_compute(n, k, uxn)
            ttv = self._many_compute(n, k, vyn)

        return ttu, ttv

    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
                carthesian: bool, cyclic_length: float, full: bool = False):

        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian,
                                        cyclic_length)

        self._elem_area = jnp.array(elem_area)
        self._dx = jnp.array(dx)
        self._dy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        self._ne_num = jnp.array(ne_num)
        self._ne_pos = jnp.array(ne_pos)
        self._area = jnp.array(area)

        smooth, metric = make_smooth(jMt, self._elem_area, self._dx, self._dy, jnn_num, jnn_pos, jtri, n2d, e2d, full)

        smooth = vmap(lambda n: smooth[:, n] / self._area[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / self._area[n])(jnp.arange(0, n2d)).T

        self._ss, self._ii, self._jj = make_smat_full(jnn_pos, jnn_num, smooth, metric, n2d, int(jnp.sum(jnn_num))) \
            if full else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        self._n2d = n2d
        self._full = full

    def prepare_from_file(self, file: str, meshtype: str = 'r', carthesian: bool = False,
                          cyclic_length: float = 360.0 * math.pi / 180.0, metric: bool = False):
        """
        Prepare the filter to be used with a mesh provided in the given file path.

        Parameters:
        -----------
        file : str
            Path to the FESOM mesh file.

        meshtype : str
        Mesh type, either 'm' (metric) or 'r' (radial). Default is radial

        carthesian : bool
            Boolean indicating whether the mesh is in Cartesian coordinates. Default is False

        cyclic_length : float
            The length of the cyclic boundary if the mesh is cyclic (for 'r' meshtype). Default is 360 * pi / 180

        metric : bool, optional
            A flag indicating whether to use the calculation including metric terms (True) or not (False).
            Default is False.
        """
        mesh = xr.open_dataset(file)
        xcoord = mesh['lon'].values
        ycoord = mesh['lat'].values

        keys = mesh.keys()
        if 'elements' in keys:
            tri = mesh['elements'].values.T - 1
        elif 'face_nodes' in keys:
            tri = mesh['face_nodes'].values.T - 1
        else:
            raise RuntimeError("In FESOM mesh file triangulation data was not found. It should be either named as elements or face_nodes")

        self.prepare(len(xcoord), len(tri[:, 1]), tri, xcoord, ycoord, meshtype, carthesian, cyclic_length, metric)
