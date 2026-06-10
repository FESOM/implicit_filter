import warnings

import jax.numpy as jnp
import numpy as np


class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


def warn_unused_gpu_argument(gpu):
    """Warn when the deprecated, ineffective ``gpu=`` argument is used.

    The flag has never had an effect; warning only when it is truthy keeps
    existing ``gpu=False`` calls (indistinguishable from the default) silent.
    """
    if gpu:
        warnings.warn(
            "the 'gpu' argument has never had an effect and is deprecated; "
            "it will be removed in a future release. Select the backend "
            "with set_backend('gpu') before the first compute instead.",
            DeprecationWarning, stacklevel=3)


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
                warnings.warn(
                    "CuPy is installed but no GPU detected, falling back to CPU."
                )
                return get_backend("cpu")

            from cupyx.scipy.sparse import csc_matrix
            from cupyx.scipy.sparse import identity
            from cupyx.scipy.sparse import diags
            from cupyx.scipy.sparse.linalg import cg as cupy_cg

            convers = cupy.asarray
            tonumpy = cupy.asnumpy

            cg = lambda Smat, ttw, x0, tol, maxiter, pre: cupy_cg(
                A=Smat, b=ttw, x0=x0, tol=tol, maxiter=maxiter, M=pre
            )
            return csc_matrix, identity, diags, cg, convers, tonumpy

        except (ImportError, RuntimeError):
            warnings.warn(
                "Requested GPU backend but CuPy is not installed. Falling back to CPU"
            )
            return get_backend("cpu")
    elif backend == "cpu":
        from scipy.sparse import csc_matrix
        from scipy.sparse import identity
        from scipy.sparse import diags
        from scipy.sparse.linalg import cg as scipy_cg
        convers = jnp.array
        tonumpy = np.array

        def cg(Smat, ttw, x0, tol, maxiter, pre):
            # scipy >= 1.12 uses rtol; older versions use tol. Try the new
            # name first and fall back, so the filter runs on both.
            try:
                return scipy_cg(A=Smat, b=ttw, x0=x0, rtol=tol, maxiter=maxiter, M=pre)
            except TypeError:
                return scipy_cg(A=Smat, b=ttw, x0=x0, tol=tol, maxiter=maxiter, M=pre)

        return csc_matrix, identity, diags, cg, convers, tonumpy
    else:
        raise NotImplementedError(f"Backend {backend} is not supported.")

def filter_stages(n: int, gamma: float) -> list[tuple[float, float]]:
    """
    Decompose an order-n implicit filter into sequential stages.

    For n >= 3 the matrix power is factorised into stages of order <= 2,
    following Danilov et al. (2024).

    Each stage is a pair (c1, c2) meaning  I + c1*Smat1 + c2*Smat1**2.
    Stages are applied in sequence (output of one is input of the next);
    their product reproduces I + gamma*Smat1**n exactly for any positive gamma.

    Parameters
    ----------
    n : int
        Filter order (1, 2, 3, or 4).
    gamma : float
        coefficient, must be positive, 2.0 for high-pass, 0.5 for low-pass.

    Returns
    -------
    list[tuple[float, float]]
        Stage coefficients; length 1 for n = 1, 2 and length 2 for n = 3, 4.

    Raises
    ------
    ValueError
        If gamma <= 0 or n not in {1, 2, 3, 4}.
    """
    if gamma <= 0:
        raise ValueError(
            f"gamma must be positive, got {gamma}"
        )

    if n == 1:
        # I + gamma * Smat1
        return [(gamma, 0.0)]

    if n == 2:
        # I + gamma * Smat1**2
        return [(0.0, gamma)]

    if n == 3:
        # 1 + gamma*x**3 = (1 + g*x)(1 - g*x + g**2 * x**2),  g = gamma**(1/3)
        g = gamma ** (1.0 / 3.0)
        return [
            (g,   0.0),       # I + g*Smat1
            (-g,  g ** 2),    # I - g*Smat1 + g**2 * Smat1**2
        ]

    if n == 4:
        # 1 + gamma*x**4 = (1 + a*x + b*x**2)(1 - a*x + b*x**2),
        #   a = (4*gamma)**(1/4), b = gamma**(1/2); the x**2 term cancels (2b - a**2 = 0).
        a = (4.0 * gamma) ** (1.0 / 4.0)
        b = gamma ** (1.0 / 2.0)
        return [
            (a,   b),    # I + a*Smat1 + b*Smat1**2
            (-a,  b),    # I - a*Smat1 + b*Smat1**2
        ]

    raise ValueError(f"filter order n must be 1, 2, 3, or 4, got {n}")

def get_gamma(highpass: bool, gamma: float | None) -> float:
    """Get gamma. Explicit gamma overrides the highpass flag."""
    if gamma is not None:
        return gamma
    return 2.0 if highpass else 0.5
