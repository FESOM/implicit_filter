from typing import List, Tuple

from ._utils import SolverNotConvergedError
from .jax_filter import JaxFilter
import jax.numpy as jnp
import numpy as np
import cupy
import concurrent
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

    def _many_compute(self, n, kl, data, tol=1e-7, maxiter=150000) -> List[np.ndarray]:
        def helper(tt: np.ndarray, i):
            with cupy.cuda.Device(i):
                tts = self._compute(n, kl, tt, tol, maxiter)
            return tts

        no_gpu = cupy.cuda.runtime.getDeviceCount()

        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for tt, i in zip(data, range(len(data))):
                futures.append(executor.submit(helper, tt, i % no_gpu))
            executor.shutdown(wait=True)

        output = []
        for f in futures:
            output.append(f.result())

        return output

    def _many_compute_full(self, n, kl, ux, vy, tol=1e-7, maxiter=150000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def helper(tt: jnp.ndarray, i):
            with cupy.cuda.Device(i):
                tts = self._compute_full(n, kl, tt, tol, maxiter)
            return tts

        no_gpu = cupy.cuda.runtime.getDeviceCount()

        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(len(ux)):
                ttuv = jnp.concatenate((ux[i], vy[i]))
                futures.append(executor.submit(helper, ttuv, i % no_gpu))
            executor.shutdown(wait=True)

        oux = []
        ovy = []
        for f in futures:
            tts = f.result()
            oux.append(tts[0:self._n2d])
            ovy.append(tts[self._n2d:2 * self._n2d])

        return oux, ovy