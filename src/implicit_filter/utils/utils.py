import math
import warnings

import jax.numpy as jnp
import numpy as np


class SolverNotConvergedError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


_PRECISION_FLOOR = 100.0   # multiples of the working epsilon


def verify_cg_convergence(apply_A, b, x, tol, maxiter, M, context, details=()):
    """
    Verify that a CG solution really reached the requested tolerance.

    JAX's ``cg`` returns ``info=None`` unconditionally -- it never reports
    failure -- so without this check a solve that ran out of iterations, or
    one whose input held NaN (where CG exits at its first iteration), is
    returned as if it were a filtered field.

    ``cg`` stops on its *recurrence* residual, which is the true residual in
    exact arithmetic but drifts away from it as the iteration proceeds, so a
    solve it declares converged can still miss ``tol`` on the residual
    recomputed here. Two thresholds separate that drift from real failure:

    * the solution is accepted at ``max(tol, _PRECISION_FLOOR * eps)``, where
      ``eps`` is the epsilon of the working dtype. A float32 solve cannot
      deliver a relative residual much below ~1e-5 however long it runs, and
      asking for more would reject solves that the scipy solver on ``main``
      returned happily; in float64 the floor is 2.2e-14 and never relaxes a
      realistic tolerance.
    * any miss beyond that gets one warm-started retry at a tenth of the
      tolerance, after which the better of the two iterates is kept -- the
      retry is not guaranteed to improve on its own starting point.

    The size of the first miss deliberately does not gate the retry. It looks
    as though it should -- a solve that is nowhere near the tolerance ought to
    be exhausted rather than drifted -- but it carries no usable signal:
    measured rescues on this package's own meshes start from 36x, 328x, 2700x
    and 19872x the tolerance, overlapping the misses that no retry can fix.
    Skipping the retry on a ratio threshold therefore rejects healthy solves,
    and the only thing it buys is a few seconds on a path that is about to
    raise anyway.

    The V-cycle path applies its own
    equivalent gate in ``_vcycle.solve_with_vcycle``, where the residual also
    has to be recomputed because that CG solves the symmetrized system
    ``(D A) x = D b`` and so measures a D-weighted residual.

    Parameters
    ----------
    apply_A : callable
        The operator the system was solved with.
    b : jnp.ndarray
        Right-hand side that was solved for.
    x : jnp.ndarray
        Solution returned by CG.
    tol, maxiter : float, int
        Tolerance and iteration cap the solve used; the retry reuses them.
    M : callable | None
        Preconditioner the solve used, reused by the retry.
    context : str
        Sentence naming the failing system, used as the message prefix.
    details : Iterable[str]
        Extra diagnostics attached to the exception.

    Returns
    -------
    jnp.ndarray
        ``x`` itself when it already meets the acceptance threshold, otherwise
        the better of ``x`` and one tighter-tolerance retry.

    Raises
    ------
    SolverNotConvergedError
        If the right-hand side is not finite, or the best iterate still misses
        the acceptance threshold.
    """
    from jax.scipy.sparse.linalg import cg

    details = list(details)
    b = jnp.asarray(b)
    b_norm = float(jnp.linalg.norm(b))
    if not math.isfinite(b_norm):
        if bool(jnp.isfinite(b).all()):
            raise SolverNotConvergedError(
                f"{context}: the magnitude of the right-hand side overflows "
                "the floating-point range, so its norm is not finite even "
                "though every element is. Rescale the data (divide by a "
                "typical magnitude) before filtering and scale the result "
                "back afterwards.",
                details)
        raise SolverNotConvergedError(
            f"{context}: the right-hand side is not finite, which happens when "
            "the input data contains NaN or inf -- or when the filter scale k "
            "does, for instance where it is zero. CG stops at its first "
            "iteration on such input and would otherwise return the data "
            "unchanged; replace non-finite values (e.g. with 0) before "
            "filtering, and use the mask argument to exclude those points.",
            details)

    # A solve can never beat the precision of the arithmetic it runs in: in
    # float32 the residual floor is ~1e-5, far above a typical tol of 1e-6.
    dtype = b.dtype
    eps = float(np.finfo(dtype if dtype.kind == "f" else np.float64).eps)
    accept = max(tol, _PRECISION_FLOOR * eps)

    def relative_residual(y):
        # An exactly zero right-hand side (a constant field, say) is solved by
        # x = 0; fall back to the absolute residual rather than dividing by 0.
        r = float(jnp.linalg.norm(b - apply_A(y)))
        return r if b_norm == 0.0 else r / b_norm

    rel = relative_residual(x)
    if rel <= accept:                   # a NaN residual fails this, as it must
        return x

    message = (
        f"{context}: CG did not reach the requested tolerance "
        f"(relative residual {{rel}} > {tol:.3e}). Stiff configurations - a "
        "high filter order n, or a filter scale far above the mesh resolution "
        "- need the multigrid preconditioner: set_preconditioner('vcycle').")

    # One bounded retry at a tighter tolerance, warm-started from the current
    # iterate. Keep whichever iterate is actually better: the retry is free to
    # come back worse, and discarding a better answer for a fresher one would
    # be a bug of its own.
    x_retry, _ = cg(apply_A, b, x0=x, tol=0.1 * tol, maxiter=maxiter, M=M)
    rel_retry = relative_residual(x_retry)
    first = rel
    if rel_retry < rel:
        x, rel = x_retry, rel_retry
    if rel <= accept:
        return x
    raise SolverNotConvergedError(
        message.format(rel=f"{rel:.3e}"), [f"first attempt: {first:.3e}", *details])


def apply_deprecated_gpu_argument(filter_obj, gpu):
    """Honour the deprecated ``gpu=`` argument of the ``prepare*`` methods.

    Before the JAX migration this argument selected the (cupy) GPU backend, so
    ignoring it silently moved such calls back onto the CPU. It now forwards to
    :meth:`Filter.set_backend`, which must happen here -- early in ``prepare``,
    after whatever argument validation that method does but before it creates
    any array -- because JAX fixes its platform on the first array created,
    which ``prepare`` itself does.

    Only a truthy value acts: ``gpu=False`` is the default and is
    indistinguishable from not passing the argument, so it must neither warn
    nor reset a backend the caller has already chosen.
    """
    if gpu:
        warnings.warn(
            "the 'gpu' argument is deprecated and will be removed in a future "
            "release; it now forwards to set_backend('gpu'). Call "
            "set_backend('gpu') explicitly before the first compute instead.",
            DeprecationWarning, stacklevel=3)
        filter_obj.set_backend("gpu")


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

