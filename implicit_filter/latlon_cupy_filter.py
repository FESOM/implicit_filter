import numpy as np
import cupy
from cupyx.scipy.sparse import csc_matrix, identity
from cupyx.scipy.sparse.linalg import cg

from implicit_filter._utils import SolverNotConvergedError
from .latlon_filter import LatLonNumpyFilter


class LatLonCupyFilter(LatLonNumpyFilter):
    def _compute(self, n, k, data: np.ndarray, maxiter=150_000, tol=1e-6) -> np.ndarray:
        e2d = self._e2d
        Smat1 = csc_matrix((cupy.asarray(self._ss) * (1.0 / k ** 2),
                            (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(e2d, e2d))
        Smat2 = identity(e2d)

        Smat = Smat2 + 0.5 * (-1 * Smat1) ** n
        ttw = cupy.asarray(data).T - Smat @ cupy.asarray(data).T

        b = 1. / Smat.diagonal()  # Simple preconditioner
        pre = csc_matrix((b, (cupy.arange(e2d), cupy.arange(e2d))), shape=(e2d, e2d))

        tts, code = cg(Smat, ttw, maxiter=maxiter, rtol=tol, M=pre)
        tts = cupy.asnumpy(tts) + data.T

        if code != 0:
            raise SolverNotConvergedError("Solver has not converged",
                                          [f"output code with code: {code}"])
        return tts