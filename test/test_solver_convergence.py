"""
The default (JAX) CG reports no status, so convergence is verified from the
true residual recomputed after the solve. These pin that a failed solve raises
instead of silently returning a wrong answer, that a successful one is
returned untouched bit for bit, and that the acceptance threshold scales with
the precision the arithmetic can actually deliver.
"""
import numpy as np
import pytest

import implicit_filter.latlon_filter as LL
import implicit_filter.triangular_filter as TF
from implicit_filter import LatLonFilter, TriangularFilter
from implicit_filter.utils._auxiliary import make_tri
from implicit_filter.utils.utils import SolverNotConvergedError, verify_cg_convergence


def cart_filter(n=10, full=False):
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    xc = np.zeros((n, n))
    yc = np.zeros((n, n))
    for i in range(n):
        yc[i, :] = np.arange(n, dtype=float)
    for i in range(n):
        xc[:, i] = np.arange(n, dtype=float)
    tri = make_tri(nodnum, n, n)
    filt = TriangularFilter()
    filt.prepare(n * n, len(tri), tri, xc.flatten(), yc.flatten(),
                 meshtype="m", cartesian=True, full=full)
    return filt, n * n


def latlon_filter():
    filt = LatLonFilter()
    filt.prepare(np.linspace(-10.0, 10.0, 8), np.linspace(0.0, 20.0, 9),
                 cartesian=True, local=True)
    return filt


def true_relative_residual(filt, n, k, data, out):
    """Residual of the perturbation system ``A tts = ttu - A ttu`` the filter
    actually solved, recomputed from the returned field ``out = tts + ttu``."""
    import jax.numpy as jnp

    kl = np.ones(len(data)) * k
    sdata = filt._ss * (1.0 / np.square(kl))[filt._jj]

    def apply_A(x):
        y = x
        for _ in range(n):
            y = jnp.zeros_like(x).at[filt._ii].add(sdata * y[filt._jj])
        return x + 2.0 * y

    ttu = jnp.asarray(data)
    b = ttu - apply_A(ttu)
    x = jnp.asarray(out) - ttu
    return float(jnp.linalg.norm(b - apply_A(x)) / jnp.linalg.norm(b))


@pytest.fixture
def gate_off(monkeypatch):
    """Return a callable that replaces the gate with the identity on demand.

    Deliberately *not* applied on entry. A test that compares the gated result
    against the ungated one has to compute the gated one first, while the real
    gate is still installed; a fixture that patched on entry would make both
    calls ungated and the comparison could never fail.
    """
    def apply_patch():
        identity = lambda apply_A, b, x, *args, **kwargs: x
        monkeypatch.setattr(TF, "verify_cg_convergence", identity)
        monkeypatch.setattr(LL, "verify_cg_convergence", identity)

    return apply_patch


class TestFailedSolveRaises:
    """A solve that cannot reach the tolerance must not return quietly."""

    def test_triangular_maxiter_exhausted(self):
        filt, n2d = cart_filter()
        data = np.random.default_rng(0).standard_normal(n2d)
        with pytest.raises(SolverNotConvergedError, match="without metric terms"):
            filt._compute(2, 1e-4, data, maxiter=1)

    def test_triangular_full_maxiter_exhausted(self):
        filt, n2d = cart_filter(full=True)
        data = np.random.default_rng(1).standard_normal(2 * n2d)
        with pytest.raises(SolverNotConvergedError, match="with metric terms"):
            filt._compute_full(2, 1e-4, data, maxiter=1)

    def test_latlon_maxiter_exhausted(self):
        filt = latlon_filter()
        data = np.random.default_rng(2).standard_normal(filt._e2d)
        with pytest.raises(SolverNotConvergedError):
            filt._compute(2, 1e-4, data, maxiter=1)

    def test_message_reports_the_achieved_residual(self):
        filt, n2d = cart_filter()
        data = np.random.default_rng(3).standard_normal(n2d)
        with pytest.raises(SolverNotConvergedError) as exc:
            filt._compute(2, 1e-4, data, maxiter=1)
        assert "relative residual" in str(exc.value)

    def test_message_names_the_remedy(self):
        filt, n2d = cart_filter()
        data = np.random.default_rng(3).standard_normal(n2d)
        with pytest.raises(SolverNotConvergedError) as exc:
            filt._compute(2, 1e-4, data, maxiter=1)
        assert "set_preconditioner('vcycle')" in str(exc.value)

    def test_errors_payload_records_the_first_attempt(self):
        # The retry always runs, so the payload always carries the residual it
        # started from -- which is what tells a stalled solve from a drifted one.
        filt, n2d = cart_filter()
        data = np.random.default_rng(0).standard_normal(n2d)
        with pytest.raises(SolverNotConvergedError) as exc:
            filt._compute(2, 1e-4, data, maxiter=1)
        assert any("first attempt" in e for e in exc.value.errors)


class TestNonFiniteInput:
    """NaN/inf used to make CG exit at once and return the input verbatim."""

    def test_nan_raises_rather_than_returning_the_input(self):
        filt, n2d = cart_filter()
        data = np.random.default_rng(4).standard_normal(n2d)
        data[7] = np.nan
        with pytest.raises(SolverNotConvergedError, match="non-finite"):
            filt.compute(1, 0.5, data)

    def test_inf_raises(self):
        filt, n2d = cart_filter()
        data = np.zeros(n2d)
        data[3] = np.inf
        with pytest.raises(SolverNotConvergedError, match="non-finite"):
            filt.compute(1, 0.5, data)

    def test_latlon_nan_raises(self):
        filt = latlon_filter()
        data = np.zeros((9, 8))
        data[2, 2] = np.nan
        with pytest.raises(SolverNotConvergedError, match="non-finite"):
            filt.compute(1, 0.5, data)

    def test_overflowing_norm_is_not_blamed_on_nan(self):
        # Every element is finite; only the norm overflows. The message must
        # say so instead of telling the user to look for a NaN they do not have.
        import jax.numpy as jnp

        b = jnp.asarray([1e300, 1e300])
        with pytest.raises(SolverNotConvergedError) as exc:
            verify_cg_convergence(lambda x: 2.0 * x, b, jnp.zeros(2),
                                  1e-6, 10, None, "ctx")
        assert "overflows" in str(exc.value)
        assert "NaN" not in str(exc.value)


class TestSuccessfulSolvesAreUntouched:
    """The gate must be invisible whenever the solve already converges."""

    def test_constant_field_is_preserved(self):
        # b is exactly zero here; a naive residual/||b|| would be 0/0 = NaN
        filt, n2d = cart_filter()
        np.testing.assert_allclose(filt.compute(1, 0.5, np.full(n2d, 3.0)), 3.0, atol=1e-10)

    def test_zero_field_is_preserved(self):
        filt, n2d = cart_filter()
        np.testing.assert_array_equal(filt.compute(1, 0.5, np.zeros(n2d)), np.zeros(n2d))

    def test_latlon_constant_field_is_preserved(self):
        filt = latlon_filter()
        np.testing.assert_allclose(filt.compute(1, 0.5, np.full((9, 8), 2.0)), 2.0, atol=1e-8)

    @pytest.mark.parametrize("n", [1, 2])
    def test_ordinary_solve_returns_a_converged_answer(self, n):
        filt, n2d = cart_filter()
        data = np.random.default_rng(5).standard_normal(n2d)
        out = filt.compute(n, 0.5, data)
        assert np.all(np.isfinite(out))
        assert true_relative_residual(filt, n, 0.5, data, out) <= 1e-6

    def test_healthy_solve_is_bit_identical_to_the_gate_off_path(self, gate_off):
        # n=1, k=0.5 converges on the first attempt, so the gate returns the
        # very same array the solver would have returned without it. The gated
        # call has to run before the patch: patch first and this compares
        # gate-off against gate-off, which cannot fail.
        filt, n2d = cart_filter()
        data = np.random.default_rng(5).standard_normal(n2d)
        gated = filt.compute(1, 0.5, data)

        filt2, _ = cart_filter()
        gate_off()
        ungated = filt2.compute(1, 0.5, data)

        np.testing.assert_array_equal(gated, ungated)

    def test_retry_rescues_a_drifted_solve(self):
        # n=2, k=0.3 on this mesh: CG stops on its recurrence residual at
        # 3.6e-5 (36x tol) while the recomputed residual still misses, and the
        # warm-started retry brings it to ~1e-7.
        n, k = 2, 0.3
        filt, n2d = cart_filter()
        data = np.random.default_rng(5).standard_normal(n2d)
        out = filt.compute(n, k, data)
        assert true_relative_residual(filt, n, k, data, out) <= 1e-6

    def test_retry_case_really_misses_on_the_first_attempt(self, gate_off):
        # Guards the test above: without the gate the same solve comes back
        # above tol, so the assertion there is exercising the retry. This one
        # deliberately never runs the real gate -- that is the whole point.
        gate_off()
        n, k = 2, 0.3
        filt, n2d = cart_filter()
        data = np.random.default_rng(5).standard_normal(n2d)
        ungated = filt.compute(n, k, data)
        assert true_relative_residual(filt, n, k, data, ungated) > 1e-6

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_float32_is_accepted_at_its_precision_floor(self, monkeypatch):
        # float32 cannot deliver a relative residual much below 1e-5, so a
        # solve that stops at ~1.3e-6 against tol=1e-6 is as converged as the
        # arithmetic allows -- scipy on main returned it, and so must we.
        n, k = 2, 0.1
        data = np.random.default_rng(11).standard_normal(2500).astype(np.float32)

        filt, _ = cart_filter(50)
        filt.set_preconditioner(None)
        gated = filt.compute(n, k, data)

        filt2, _ = cart_filter(50)
        filt2.set_preconditioner(None)
        monkeypatch.setattr(TF, "verify_cg_convergence",
                            lambda apply_A, b, x, *a, **kw: x)
        ungated = filt2.compute(n, k, data)

        np.testing.assert_array_equal(gated, ungated)


class TestHelper:
    """Unit-level behaviour of the gate itself."""

    def apply(self, x):
        import jax.numpy as jnp
        return 2.0 * jnp.asarray(x)

    def test_returns_the_solution_when_converged(self):
        import jax.numpy as jnp
        b = jnp.asarray([2.0, 4.0])
        x = jnp.asarray([1.0, 2.0])                      # exact solution
        out = verify_cg_convergence(self.apply, b, x, 1e-6, 10, None, "ctx")
        np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_zero_rhs_with_zero_solution_is_converged(self):
        import jax.numpy as jnp
        z = jnp.zeros(3)
        out = verify_cg_convergence(self.apply, z, z, 1e-6, 10, None, "ctx")
        np.testing.assert_array_equal(np.asarray(out), np.asarray(z))

    def test_non_finite_rhs_raises_immediately(self):
        import jax.numpy as jnp
        b = jnp.asarray([np.nan, 1.0])
        with pytest.raises(SolverNotConvergedError, match="non-finite"):
            verify_cg_convergence(self.apply, b, jnp.zeros(2), 1e-6, 10, None, "ctx")

    def test_context_and_details_reach_the_exception(self):
        import jax.numpy as jnp
        b = jnp.asarray([np.nan])
        with pytest.raises(SolverNotConvergedError) as exc:
            verify_cg_convergence(self.apply, b, jnp.zeros(1), 1e-6, 10, None,
                                  "my context", details=["k=3"])
        assert "my context" in str(exc.value)
        assert "k=3" in exc.value.errors

    # maxiter=0 makes the retry a no-op, so both of these test the acceptance
    # threshold alone rather than the solvability of this toy operator.
    def test_float32_rhs_relaxes_the_acceptance_to_the_precision_floor(self):
        # rel = 2e-6: far above tol, but well inside the float32 floor of
        # 100 * eps = 1.2e-5, so the iterate is accepted unchanged.
        import jax.numpy as jnp
        b = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
        x = jnp.asarray([0.5, 0.5 * (1.0 + 4e-6)], dtype=jnp.float32)
        out = verify_cg_convergence(self.apply, b, x, 1e-9, 0, None, "ctx")
        np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_float64_rhs_does_not_get_the_relaxation(self):
        # The identical residual in float64, where 100 * eps = 2.2e-14 leaves
        # tol in charge, is a failure.
        import jax.numpy as jnp
        b = jnp.asarray([1.0, 1.0])
        x = jnp.asarray([0.5, 0.5 * (1.0 + 4e-6)])
        with pytest.raises(SolverNotConvergedError):
            verify_cg_convergence(self.apply, b, x, 1e-9, 0, None, "ctx")
