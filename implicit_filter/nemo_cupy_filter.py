import numpy as np
import cupy
from cupyx.scipy.sparse import csc_matrix
from cupyx.scipy.sparse.linalg import cg

from implicit_filter._utils import SolverNotConvergedError
from .nemo_filter import NemoNumpyFilter


class NemoCupyFilter(NemoNumpyFilter):
    def _compute(self, n: int, k: float, data: np.ndarray, maxiter: int = 150_000, tol: float = 1e-6) -> np.ndarray:
        Smat1 = csc_matrix((cupy.asarray(self._ss) * (1.0 / k ** 2), (cupy.asarray(self._ii), cupy.asarray(self._jj))),
                           shape=(self._e2d, self._e2d))
        Smat2 = csc_matrix((cupy.asarray(self._area), (cupy.arange(self._e2d), cupy.arange(self._e2d))),
                           shape=(self._e2d, self._e2d))

        cdata = cupy.asarray(data)

        Smat = Smat2 - 0.5 * Smat1 ** n
        ttw = (cdata * cupy.asarray(self._area)) - Smat @ cdata  # Work with perturbations

        # b = 1. / Smat.diagonal()  # Simple preconditioner
        # pre = csc_matrix((b, (np.arange(self._e2d), np.arange(self._e2d))), shape=(self._e2d, self._e2d))

        tts, code = cg(Smat, ttw, maxiter=maxiter, tol=tol)
        tts += cdata

        if code != 0:
            raise SolverNotConvergedError("Solver has not converged",
                                          [f"output code with code: {code}"])

        return tts.get()
