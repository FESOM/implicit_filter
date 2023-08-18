from .jax_filter import JaxFilter
import numpy as np
import cupy
from cupyx.scipy.sparse import csc_matrix as cupy_csc
from cupyx.scipy.sparse import identity as cupy_identity
from cupyx.scipy.sparse.linalg import cg as cupy_cg


class CuPyFilter(JaxFilter):
    """
    Class uses CuPy for sparse matrix algebra and solver instead of SciPy

    This class still uses JAX for preparing auxiliary arrays on CPU
    """
    def _compute(self, n, kl, ttu, tol=1e-6, maxiter=150000):
        ttu = cupy.asarray(ttu)
        Smat1 = cupy_csc((cupy.asarray(self._ss) * (1.0 / cupy.square(kl)),
                          (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(self._n2d, self._n2d))
        Smat = cupy_identity(self._n2d) + 0.5 * (Smat1 ** n)
        # ===============
        # Solve with conjugate gradient
        # ===============
        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cupy_cg(Smat, ttw, tol=1e-6, maxiter=150000)
        if code != 0:
            print(code)
        tts += ttu
        return tts.get()

    def _compute_full(self, n, kl, ttuv, tol=1e-6, maxiter=150000):
        ttuv = cupy.asarray(ttuv)
        Smat1 = cupy_csc((cupy.asarray(self._ss) * (1.0 / cupy.square(kl)),
                          (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = cupy_identity(2 * self._n2d) + 0.5 * (Smat1 ** n)
        # ===============
        # Solve with conjugate gradient
        # ===============
        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cupy_cg(Smat, ttw, tol=1e-7, maxiter=150000)
        if code != 0:
            print(code)
        tts += ttuv
        return tts.get()
