import math
from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg

from implicit_filter._auxiliary import neighbouring_nodes, neighboring_triangles, areas
from implicit_filter._numpy_functions import make_smooth, make_smat
from implicit_filter._utils import VeryStupidIdeaError, SolverNotConvergedError
from implicit_filter.filter import Filter


class NumpyFilter(Filter):
    """
    A class for filtering data using JAX-based implicit filtering techniques.
    Extends the base Filter class.
    """

    def _check_filter_order(self, n: int) -> None:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        self._check_filter_order(n)
        return self._compute(n, k, data)

    def compute_velocity(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Filtering non-scalar values are not supported with NumPy Filer")

    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str = 'r',
                carthesian: bool = False, cyclic_length: float = 360.0 * math.pi / 180.0, full: bool = False):
        if full:
            raise NotImplementedError("Computation including metric terms are not supported with NumPy Filer")

        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian,
                                            cyclic_length)

        self._elem_area = elem_area
        self._dx = dx
        self._dy = dy
        self._ne_num = ne_num
        self._ne_pos = ne_pos
        self._area = area

        smooth = make_smooth(self._elem_area, self._dx, self._dy, nn_num, nn_pos, tri, n2d, e2d)

        for i in range(n2d):
            smooth[:, i] /= self._area[i]

        self._ss, self._ii, self._jj = make_smat(nn_pos, nn_num, smooth, n2d, int(np.sum(nn_num)))
        self._n2d = n2d
        self._full = full

    def __transform_atribute(self, atr: str, lmbd, fill=None):
        """
        If atribute atr exists then transform it using given Callable lmbd, otherwise it set with fill value
        """
        if hasattr(self, atr):
            setattr(self, atr, lmbd(getattr(self, atr)))
        else:
            setattr(self, atr, fill)

    def __init__(self, *initial_data, **kwargs):
        super().__init__(initial_data, kwargs)
        # Transform from Numpy array
        self.__transform_atribute("_n2d", lambda x: int(x), 0)
        self.__transform_atribute("_full", lambda x: bool(x), False)

    def _compute(self, n, kl, ttu, tol=1e-6, maxiter=150000):
        Smat1 = csc_matrix((self._ss * (1.0 / np.square(kl)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)

        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged without metric terms",
                                          [f"output code with code: {code}"])

        tts += ttu
        return np.array(tts)
