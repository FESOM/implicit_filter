import numpy as np
import jax.numpy as jnp
from jax import vmap
from ._auxiliary import neighboring_triangles, neighbouring_nodes, areas
from ._jax_function import make_smooth, make_smat, make_smat_full
from implicit_filter.filter import Filter
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg


class JaxFilter(Filter):
    """
        A class that implements filtering using JAX for implicit filtering techniques.

        This class inherits from the abstract base class Filter and provides methods for computing filtered data,
        preparing filters for specific meshes, and conducting filtering using JAX.

        Parameters:
        ----------
        Filter : ABC
            Abstract base class for filter implementations.

        Attributes:
        -----------
        __ss : jnp.ndarray
            Private attribute storing the nonzero entries of a sparse matrix.

        __ii : jnp.ndarray
            Private attribute storing the row indices of the nonzero entries.

        __jj : jnp.ndarray
            Private attribute storing the column indices of the nonzero entries.

        __n2d : int
            Private attribute storing the total number of nodes in the mesh.

        __full : bool
            Private attribute indicating whether to use metric terms when filtering

        Methods:
        --------
        compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
            Compute filtered data using JAX-based implicit filtering techniques.

        prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray,
                meshtype: str, carthesian: bool, full: bool, cyclic_length):
            Prepare the filter for a specific mesh and type of filtering.

        Notes:
        ------
        This class assumes that the Filter abstract base class is defined. It utilizes JAX for efficient filtering
        computations.
    """

    __ss: jnp.ndarray
    __ii: jnp.ndarray
    __jj: jnp.ndarray

    __n2d: int
    __full: bool

    def __compute(self, n, kl, ttu, tol=1e-6, maxiter=150000):
        Smat1 = csc_matrix((self.__ss * (1.0 / jnp.square(kl)), (self.__ii, self.__jj)), shape=(self.__n2d, self.__n2d))
        Smat = identity(self.__n2d) + 0.5 * (Smat1 ** n)

        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            print(code)
        tts += ttu
        return tts

    def __compute_full(self, n, kl, ttuv, tol=1e-6, maxiter=150000):
        Smat1 = csc_matrix((self.__ss * (1.0 / jnp.square(kl)), (self.__ii, self.__jj)), shape=(2 * self.__n2d, 2 * self.__n2d))
        Smat = identity(2 * self.__n2d) + 0.5 * (Smat1 ** n)

        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            print(code)
        tts += ttuv
        return tts

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        return np.array(self.__compute_full(n, k, data) if self.__full else self.__compute(n, k, data))

    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
                carthesian: bool, cyclic_length: float, full: bool = False):
        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian,
                                        cyclic_length)
        jelem_area = jnp.array(elem_area)
        jdx = jnp.array(dx)
        jdy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        jarea = jnp.array(area)

        smooth, metric = make_smooth(jMt, jelem_area, jdx, jdy, jnn_num, jnn_pos, jtri, n2d, e2d, False)

        smooth = vmap(lambda n: smooth[:, n] / jarea[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / jarea[n])(jnp.arange(0, n2d)).T

        self.__ss, self.__ii, self.__jj = make_smat_full(jnn_pos, jnn_num, smooth, jMt, n2d, int(jnp.sum(jnn_num))) if full \
            else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        self.__n2d = n2d
        self.__full = full
