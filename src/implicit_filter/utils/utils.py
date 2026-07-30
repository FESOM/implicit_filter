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

