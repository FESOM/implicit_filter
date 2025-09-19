import warnings

import jax.numpy as jnp
import numpy as np

class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class VeryStupidIdeaError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class TheHollyHandErrorOfAntioch(Exception):
    def __init__(self):
        message = "Then shalt thou count to two, no more, no less. Two shall be the number thou shalt filter, and the number of the filter shall be two."
        super().__init__(message)
        self.errors = ["Three shalt thou not count,"]

class SizeMissmatchError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

def transform_attribute(self, atr: str, lmbd, fill=None):
    """
    If attribute atr exists, then transform it using given Callable lmbd; otherwise it set with fill value
    """
    if hasattr(self, atr):
        setattr(self, atr, lmbd(getattr(self, atr)))
    else:
        setattr(self, atr, fill)


def get_backend(backend: str):
    if backend == "gpu":
        try:
            import cupy
            
            # Check if at least one GPU device is available
            if cupy.cuda.runtime.getDeviceCount() == 0:
                warnings.warn("CuPy is installed but no GPU detected, falling back to CPU.")
                return get_backend("cpu")
            
            from cupyx.scipy.sparse import csc_matrix
            from cupyx.scipy.sparse import identity
            from cupyx.scipy.sparse.linalg import cg as cupy_cg
            convers = cupy.asarray
            tonumpy = cupy.asnumpy

            cg = lambda Smat, ttw, x0, tol, maxiter, pre: cupy_cg(A=Smat, b=ttw, x0=x0, tol=tol, maxiter=maxiter, M=pre)
            return csc_matrix, identity, cg, convers, tonumpy

        except (ImportError, RuntimeError):
            warnings.warn("Requested GPU backend but CuPy is not installed. Falling back to CPU")
            return get_backend("cpu")
    elif backend == "cpu":
        from scipy.sparse import csc_matrix
        from scipy.sparse import identity
        from scipy.sparse.linalg import cg as scipy_cg
        convers = jnp.array
        tonumpy = np.array

        cg = lambda Smat, ttw, x0, tol, maxiter, pre: scipy_cg(A=Smat, b=ttw, x0=x0, rtol=tol, maxiter=maxiter, M=pre)
        return csc_matrix, identity, cg, convers, tonumpy
    else:
        raise NotImplementedError(f"Backend {backend} is not supported.")
    
    