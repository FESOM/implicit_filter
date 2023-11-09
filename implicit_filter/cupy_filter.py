from typing import List

from ._utils import SolverNotConvergedError
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
    def _compute(self, n, kl, ttu, tol=1e-6, maxiter=150000) -> np.ndarray:
        ttu = cupy.asarray(ttu)
        Smat1 = cupy_csc((cupy.asarray(self._ss) * (1.0 / cupy.square(kl)),
                          (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(self._n2d, self._n2d))
        Smat = cupy_identity(self._n2d) + 0.5 * (Smat1 ** n)
        # ===============
        # Solve with conjugate gradient
        # ===============
        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cupy_cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged without metric terms",
                                          [f"output code with code: {code}"])

        tts += ttu
        return tts.get()

    def _compute_full(self, n, kl, ttuv, tol=1e-6, maxiter=150000) -> np.ndarray:
        ttuv = cupy.asarray(ttuv)
        Smat1 = cupy_csc((cupy.asarray(self._ss) * (1.0 / cupy.square(kl)),
                          (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = cupy_identity(2 * self._n2d) + 0.5 * (Smat1 ** n)
        # ===============
        # Solve with conjugate gradient
        # ===============
        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cupy_cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged with metric terms",
                                          [f"output code with code: {code}"])

        tts += ttuv
        return tts.get()

    def _many_compute(self, n, kl, data, tol=1e-6, maxiter=150000) -> List[np.ndarray]:
        Smat1 = cupy_csc((cupy.asarray(self._ss) * (1.0 / cupy.square(kl)),
                          (cupy.asarray(self._ii), cupy.asarray(self._jj))), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = cupy_identity(2 * self._n2d) + 0.5 * (Smat1 ** n)

        no_gpu = cupy.cuda.runtime.getDeviceCount()
        output = []
        for tt, i in zip(data, range(len(data))):
            with cupy.cuda.Device(i % no_gpu):
                ttu = cupy.asarray(tt)
                ttw = ttu - Smat @ ttu  # Work with perturbations

                tts, code = cupy_cg(Smat, ttw, tol=tol, maxiter=maxiter)
                if code != 0:
                    raise SolverNotConvergedError("Solver has not converged with metric terms",
                                                  [f"output code with code: {code}"])

                output.append(tts)

        return output