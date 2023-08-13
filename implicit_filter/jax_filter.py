from typing import Tuple, Optional

import numpy as np
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import vmap, jit
from ._auxiliary import neighboring_triangles, neighbouring_nodes, areas
from ._jax_function import make_smooth, make_smat, make_smat_full, transform_veloctiy_to_nodes
from implicit_filter.filter import Filter
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg


class JaxFilter(Filter):
    def __init__(self, *initial_data, **kwargs):
        super().__init__(initial_data, kwargs)
        self.__elem_area: Optional[jnp.ndarray] = None
        self.__area: Optional[jnp.ndarray] = None
        self.__ne_pos: Optional[jnp.ndarray] = None
        self.__ne_num: Optional[jnp.ndarray] = None
        self.__dx: Optional[jnp.ndarray] = None
        self.__dy: Optional[jnp.ndarray] = None

        self.__ss: Optional[jnp.ndarray] = None
        self.__ii: Optional[jnp.ndarray] = None
        self.__jj: Optional[jnp.ndarray] = None

        self.__n2d: int = 0
        self.__full: bool = False

    def __compute(self, n, kl, ttu, tol=1e-6, maxiter=150000):
        Smat1 = csc_matrix((self.__ss * (1.0 / jnp.square(kl)), (self.__ii, self.__jj)), shape=(self.__n2d, self.__n2d))
        Smat = identity(self.__n2d) + 0.5 * (Smat1 ** n)

        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            print(code)
        tts += ttu
        return np.array(tts)

    def __compute_full(self, n, kl, ttuv, tol=1e-6, maxiter=150000):
        Smat1 = csc_matrix((self.__ss * (1.0 / jnp.square(kl)), (self.__ii, self.__jj)), shape=(2 * self.__n2d, 2 * self.__n2d))
        Smat = identity(2 * self.__n2d) + 0.5 * (Smat1 ** n)

        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            print(code)
        tts += ttuv
        return np.array(tts)

    def compute_velocity(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uxn, vyn = transform_veloctiy_to_nodes(jnp.array(ux), jnp.array(vy), self.__ne_pos, self.__ne_num, self.__n2d,
                                               self.__elem_area, self.__area)
        if self.__full:
            ttuv = self.__compute_full(n, k, jnp.concatenate((uxn, vyn)))
            return ttuv[0:self.__n2d], ttuv[self.__n2d:2*self.__n2d]
        else:
            ttu = self.__compute(n, k, uxn)
            ttv = self.__compute(n, k, vyn)
            return ttu, ttv

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        return np.array(self.__compute_full(n, k, data) if self.__full else self.__compute(n, k, data))

    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
                carthesian: bool, cyclic_length: float, full: bool = False):
        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian,
                                        cyclic_length)

        self.__elem_area = jnp.array(elem_area)
        self.__dx = jnp.array(dx)
        self.__dy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        self.__ne_num = jnp.array(ne_num)
        self.__ne_pos = jnp.array(ne_pos)
        self.__area = jnp.array(area)

        smooth, metric = make_smooth(jMt, self.__elem_area, self.__dx, self.__dy, jnn_num, jnn_pos, jtri, n2d, e2d, False)

        smooth = vmap(lambda n: smooth[:, n] / self.__area[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / self.__area[n])(jnp.arange(0, n2d)).T

        self.__ss, self.__ii, self.__jj = make_smat_full(jnn_pos, jnn_num, smooth, metric, n2d, int(jnp.sum(jnn_num))) \
            if full else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        self.__n2d = n2d
        self.__full = full
