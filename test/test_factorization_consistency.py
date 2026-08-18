"""
Consistency tests for the factorized higher-order filter and variable gamma.

Verifies:
  1. n=1,2 with gamma=2.0 reproduces the original  identity + 2.0*(Smat1**n)  formula
     (backward compatibility for existing users).
  2. The stage factorization for n=3,4 reproduces  identity + gamma*(Smat1**n)
     at the matrix level, for several gamma including a non-standard value.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator

from implicit_filter import TriangularFilter
from implicit_filter.utils._auxiliary import make_tri
from implicit_filter.utils.utils import filter_stages


# ---------------------------------------------------------------------------
# Shared mesh builder (same style as the existing integration tests)
# ---------------------------------------------------------------------------

def build_filter(Lx=20, filter_elements=True):
    xx = np.arange(0, Lx + 1, dtype=float)
    yy = np.arange(0, Lx + 1, dtype=float)
    nx, ny = len(xx), len(yy)
    nodnum = np.reshape(np.arange(nx * ny), [ny, nx]).T
    xcoord = np.zeros((nx, ny))
    ycoord = xcoord.copy()
    for i in range(nx):
        ycoord[i, :] = yy
    for i in range(ny):
        xcoord[:, i] = xx
    xcoord = xcoord.flatten()
    ycoord = ycoord.flatten()
    tri = make_tri(nodnum, nx, ny)
    n2d, e2d = len(xcoord), len(tri)
    filt = TriangularFilter()
    filt.prepare(
        n2d, e2d, tri, xcoord, ycoord,
        meshtype='m', cartesian=True, full=False,
        filter_elements=filter_elements,
    )
    return filt, n2d, e2d


# ---------------------------------------------------------------------------
# Reference helpers: rebuild the OLD behavior independently inside the test
#
# These used to go through the filter's backend abstraction
# (filt.csc_matrix / convers / identity / cg / tonumpy). That abstraction was
# removed when TriangularFilter went matrix-free; on the CPU backend it was
# scipy anyway, so the reference is now written directly against scipy.
# ---------------------------------------------------------------------------

def _reference_cg(A, b, M, tol=1e-6, maxiter=150000):
    """scipy CG, tolerant of the tol -> rtol rename in scipy 1.14."""
    try:
        return cg(A, b, M=M, tol=tol, maxiter=maxiter)
    except TypeError:
        return cg(A, b, M=M, rtol=tol, maxiter=maxiter)


def _build_scaled_smat1(filt, ss, ii, jj, n_size, k):
    """Build and 1/k**2-scale Smat1 as _compute does.

    Column-wise scaling is kept (rather than a single scalar multiply) so
    that a spatially varying k array would work here too.
    """
    kl = np.ones(n_size) * k
    Smat1 = sp.csc_matrix(
        (np.asarray(ss, dtype=np.float64),
         (np.asarray(ii), np.asarray(jj))),
        shape=(n_size, n_size),
    )
    scaling_vector = 1.0 / np.square(kl)
    nnz_per_column = np.diff(Smat1.indptr)
    multipliers = np.repeat(scaling_vector, nnz_per_column)
    Smat1.data = Smat1.data * multipliers
    return Smat1


def _old_compute(filt, n, k, data):
    """Original single-system formula:  identity + 2.0*(Smat1**n).

    Uses Smat1**n (NOT @) because ** is the true sparse matrix power here.
    """
    n_size = len(data)
    is_elem = (n_size == filt._e2d)
    ss = filt._ss_e if is_elem else filt._ss
    ii = filt._ii_e if is_elem else filt._ii
    jj = filt._jj_e if is_elem else filt._jj

    Smat1 = _build_scaled_smat1(filt, ss, ii, jj, n_size, k)
    Smat = (sp.identity(n_size, format="csc") + 2.0 * (Smat1 ** n)).tocsc()

    ttu = np.asarray(data, dtype=np.float64)
    ttw = ttu - Smat @ ttu

    b = 1.0 / Smat.diagonal()  # Jacobi preconditioner
    pre = LinearOperator(Smat.shape, matvec=lambda x, b=b: b * x)

    tts, code = _reference_cg(Smat, ttw, pre)
    assert code == 0, "reference solver did not converge"
    return np.asarray(tts + ttu)


# ---------------------------------------------------------------------------
# 1. Backward compatibility: n=1,2 with gamma=2.0 unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """New _compute (gamma default 2.0) must match the old formula for n=1,2."""

    @pytest.mark.parametrize("n", [1, 2])
    def test_scalar_nodes_match_old(self, n):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)

        new = filt.compute(n, 5.0, data)          # gamma default -> 2.0 (high-pass)
        old = _old_compute(filt, n, 5.0, data)

        np.testing.assert_allclose(new, old, atol=1e-8,
            err_msg=f"n={n} nodal scalar differs from old formula")

    @pytest.mark.parametrize("n", [1, 2])
    def test_scalar_elements_match_old(self, n):
        filt, _, e2d = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(e2d)

        new = filt.compute(n, 5.0, data)
        old = _old_compute(filt, n, 5.0, data)

        np.testing.assert_allclose(new, old, atol=1e-7,
            err_msg=f"n={n} element scalar differs from old formula")


# ---------------------------------------------------------------------------
# 2. Factorization correctness for n=3,4 at the matrix level
# ---------------------------------------------------------------------------

class TestFactorization:
    """Product of stage operators must equal  identity + gamma*(Smat1**n)."""

    @pytest.mark.parametrize("n", [3, 4])
    @pytest.mark.parametrize("gamma", [2.0, 0.5, 1.7])
    def test_stage_product_matches_target(self, n, gamma):
        filt, n2d, _ = build_filter(20)
        Smat1 = _build_scaled_smat1(filt, filt._ss, filt._ii, filt._jj, n2d, 5.0)
        I = sp.identity(n2d, format="csc")

        # target operator
        target = (I + gamma * (Smat1 ** n)).toarray()

        # product of stages (** is the true sparse matrix power)
        prod = I
        for (c1, c2) in filter_stages(n, gamma):
            stage = I + c1 * Smat1
            if c2 != 0.0:
                stage = stage + c2 * (Smat1 ** 2)
            prod = prod @ stage

        prod = prod.toarray()
        diff = np.max(np.abs(prod - target))
        print(f"n={n}, gamma={gamma}: max abs diff = {diff:.2e}")
        np.testing.assert_allclose(prod, target, atol=1e-7,
            err_msg=f"stage product != target for n={n}, gamma={gamma}, diff {diff:.2e}")


# ---------------------------------------------------------------------------
# 3. gamma / highpass plumbing
# ---------------------------------------------------------------------------

class TestGammaPlumbing:
    """Explicit gamma and the highpass flag reach the operator correctly."""

    def test_highpass_default_equals_gamma_2(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)

        default = filt.compute(2, 5.0, data)                      # default highpass
        explicit = filt.compute(2, 5.0, data, gamma=2.0)          # explicit gamma=2
        np.testing.assert_allclose(default, explicit, atol=1e-10,
            err_msg="default call does not equal explicit gamma=2.0")

    def test_lowpass_differs_from_highpass(self):
        filt, n2d, _ = build_filter(20)
        np.random.seed(42)
        data = np.random.randn(n2d)

        hp = filt.compute(2, 5.0, data, gamma=2.0)
        lp = filt.compute(2, 5.0, data, gamma=0.5)
        assert not np.allclose(hp, lp), \
            "gamma=2.0 and gamma=0.5 should give different results"