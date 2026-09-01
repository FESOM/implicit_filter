import math
import warnings
from typing import Tuple, Iterable

import numpy as np

from implicit_filter.utils._auxiliary import R_EARTH
from implicit_filter.utils._numpy_functions import (
    calculate_local_regular_neighbourhood,
    calculate_global_regular_neighbourhood,
)
from implicit_filter.filter import Filter
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    transform_attribute,
    warn_unused_gpu_argument,
    filter_stages,
    get_gamma,
)
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def _offdiag_rel_asym(off_ii, off_jj, off_ss, weight, n2d):
    """Relative asymmetry of ``D @ S`` restricted to off-diagonal entries."""
    import scipy.sparse as sp

    K = sp.csr_matrix((weight[off_ii] * off_ss, (off_ii, off_jj)), shape=(n2d, n2d))
    diff = (K - K.T).tocoo()
    if diff.nnz == 0:
        return 0.0
    denom = max(np.abs(K.data).max(), 1e-300)
    return float(np.abs(diff.data).max() / denom)


def _symmetrize_stencil(ss, ii, jj, area):
    """Force ``D @ S`` to be exactly symmetric, choosing ``D`` automatically.

    Unlike :class:`TriangularFilter`'s edge-based assembly (each edge weight
    is computed once and shared, so its ``D @ S`` is symmetric to the bit),
    this class recomputes each edge's geometry independently from both
    endpoints (once from node ``n``'s perspective, once from its neighbor's).
    Those two computations agree only approximately, and how closely depends
    on the grid: a spherical grid (``cartesian=False``, zonal spacing scaled
    by cos(latitude)) is symmetric under ``D = diag(area)``; a stretched but
    unscaled tensor-product Cartesian grid (``cartesian=True``, ``hh``
    separable in x and y) is instead symmetric under ``D = diag(area**2)``
    -- each is exact for one grid family and badly wrong for the other (the
    docstring in the vcycle call sites below has the derivation). Rather
    than hardcode one, pick whichever weight already nearly-symmetrizes
    ``D @ S`` for this particular grid.

    Either way the two computations only agree to float64 roundoff
    (~1e-16 relative), and squaring the stencil for a quadratic filter stage
    amplifies that residual by orders of magnitude through cancellation, to
    the point of tripping the V-cycle preconditioner's hard symmetry gate
    (measured ~1e-3 on a plain uniform lat-lon box) even though the
    underlying geometry is symmetric. Averaging each off-diagonal pair under
    the chosen weight removes that roundoff seed at the source, before any
    stage matrix is ever formed. The diagonal is rebuilt from the
    symmetrized off-diagonals (rather than symmetrized itself) to preserve
    ``S @ ones == 0``, i.e. that a constant field is unaffected by the
    filter, and is only rebuilt for nodes that already had entries (masked
    -- land -- nodes must stay entirely absent from the stencil).

    Returns
    -------
    (ss, ii, jj, weight) : the symmetrized stencil plus the chosen ``D``
        weight (``area`` or ``area**2``), for reuse by the V-cycle call
        sites so they don't have to re-derive which one this grid needs.
    """
    import scipy.sparse as sp

    n2d = len(area)
    valid_nodes = np.unique(ii)
    diag_mask = ii == jj
    off_ii, off_jj, off_ss = ii[~diag_mask], jj[~diag_mask], ss[~diag_mask]

    area2 = np.square(area)
    weight = (area if _offdiag_rel_asym(off_ii, off_jj, off_ss, area, n2d)
                      <= _offdiag_rel_asym(off_ii, off_jj, off_ss, area2, n2d)
              else area2)

    K = sp.csr_matrix((weight[off_ii] * off_ss, (off_ii, off_jj)), shape=(n2d, n2d))
    K_sym = ((K + K.T) * 0.5).tocoo()

    sym_ii, sym_jj = K_sym.row, K_sym.col
    sym_ss = K_sym.data / weight[sym_ii]

    row_sum = np.zeros(n2d, dtype=sym_ss.dtype)
    np.add.at(row_sum, sym_ii, sym_ss)

    out_ii = np.concatenate([sym_ii, valid_nodes])
    out_jj = np.concatenate([sym_jj, valid_nodes])
    out_ss = np.concatenate([sym_ss, -row_sum[valid_nodes]])
    return out_ss, out_ii, out_jj, weight


class LatLonFilter(Filter):
    """
    Filter implementation for regular latitude-longitude grids.

    This class provides implicit filtering capabilities for data on structured
    lat-lon grids. It supports both Cartesian and spherical coordinate systems
    with configurable boundary conditions and land-sea masks.

    Parameters
    ----------
    See Filter class for inherited parameters.

    Attributes
    ----------
    _n2d : int
        Total number of grid points (nx * ny)
    _nx : int
        Number of longitude points
    _ny : int
        Number of latitude points
    _ss : np.ndarray
        Non-zero values of sparse filter matrix
    _ii : np.ndarray
        Row indices for sparse matrix entries
    _jj : np.ndarray
        Column indices for sparse matrix entries
    _area : np.ndarray
        Area associated with each grid cell

    _mask_n : np.ndarray
        Boolean mask for valid grid points (False indicates land)
    """

    def __init__(self, *initial_data, **kwargs):
        super().__init__(*initial_data, **kwargs)
        it = lambda ar: int(ar)
        ar = lambda ar: np.array(ar)
        st = lambda ar: str(ar)

        # Files saved before the _e2d -> _n2d rename restore as _e2d; adopt
        # it so transform_attribute below doesn't reset _n2d to the fill value.
        if hasattr(self, "_e2d") and not hasattr(self, "_n2d"):
            self._n2d = self._e2d

        # Transform and initialize attributes with default values
        transform_attribute(self, "_n2d", it, 0)
        transform_attribute(self, "_nx", it, 0)
        transform_attribute(self, "_ny", it, 0)
        transform_attribute(self, "_ss", ar, None)
        transform_attribute(self, "_ii", ar, None)
        transform_attribute(self, "_jj", ar, None)
        transform_attribute(self, "_area", ar, None)

    def prepare(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        cartesian: bool = False,
        local: bool = True,
        cyclic_length: float = 2 * math.pi,
        mask: np.ndarray | None = None,
        gpu: bool = False,
    ):
        """
        Configure filter for a latitude-longitude grid.

        Computes grid topology, geometric properties, and assembles the filter
        operator matrix. Must be called before any filtering operations.

        Parameters
        ----------
        latitude : np.ndarray
            Latitude values in degrees (1D array)
        longitude : np.ndarray
            Longitude values in degrees (1D array)
        cartesian : bool, optional
            True for Cartesian coordinates, False for spherical (default)
        local : bool, optional
            True for 4-point local neighborhood, False for 8-point global (default: True)
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        mask : np.ndarray, optional
            Land-sea mask where True indicates land (default: all ocean)
        gpu : bool, optional
            Deprecated and without effect; select the backend with
            :meth:`set_backend` instead.

        Notes
        -----
        - Land points are masked using Neumann boundary conditions
        """
        warn_unused_gpu_argument(gpu)
        nx = len(longitude)
        ny = len(latitude)
        n2d = nx * ny
        self._n2d = n2d
        self._nx = nx
        self._ny = ny

        xcoord = np.zeros((nx, ny))
        ycoord = xcoord.copy()

        for i in range(nx):
            ycoord[i, :] = latitude

        for i in range(ny):
            xcoord[:, i] = longitude

        xcoord = np.reshape(xcoord, [nx * ny])
        ycoord = np.reshape(ycoord, [nx * ny])

        self._mask_n = (
            np.ones(self._n2d, dtype=bool) if mask is None else mask.flatten()
        )

        if local:
            ee_pos, nza = calculate_local_regular_neighbourhood(n2d, nx, ny)
        else:
            ee_pos, nza = calculate_global_regular_neighbourhood(n2d, nx, ny)

        rad = math.pi / 180.0

        if cartesian:
            Mt = np.ones(n2d)
        else:
            Mt = np.cos(np.sum(rad * ycoord[ee_pos], axis=0) / 4.0)

        hh = np.ones((4, n2d))  # Edge lengths
        hc = np.ones((4, n2d))  # Distance to next cell centers
        r_earth = R_EARTH

        # Fill ee_pos, arrangement is W;N;E;S
        for i in range(n2d):
            if ee_pos[1, i] == i:
                hc[1, i] = rad * r_earth * (ycoord[i] - ycoord[ee_pos[3, i]])  # S
            else:
                hc[1, i] = rad * r_earth * (ycoord[ee_pos[1, i]] - ycoord[i])  # N

            if ee_pos[3, i] == i:
                hc[3, i] = rad * r_earth * (ycoord[ee_pos[1, i]] - ycoord[i])  # N
            else:
                hc[3, i] = rad * r_earth * (ycoord[i] - ycoord[ee_pos[3, i]])  # S

            if ee_pos[0, i] == i:
                hc[0, i] = rad * (xcoord[ee_pos[2, i]] - xcoord[i])  # E
            else:
                hc[0, i] = rad * (xcoord[i] - xcoord[ee_pos[0, i]])  # W

            if ee_pos[2, i] == i:
                hc[2, i] = rad * (xcoord[i] - xcoord[ee_pos[0, i]])  # W
            else:
                hc[2, i] = rad * (xcoord[ee_pos[2, i]] - xcoord[i])  # E

            if hc[0, i] > cyclic_length / 2.0:
                hc[0, i] = hc[0, i] - cyclic_length
            if hc[0, i] < -cyclic_length / 2.0:
                hc[0, i] = hc[0, i] + cyclic_length

            if hc[2, i] > cyclic_length / 2.0:
                hc[2, i] = hc[2, i] - cyclic_length
            if hc[2, i] < -cyclic_length / 2.0:
                hc[2, i] = hc[2, i] + cyclic_length

            hc[0, i] *= Mt[i] * r_earth
            hc[2, i] *= Mt[i] * r_earth

        hh[1, :] = (hc[1, :] + hc[3, :]) / 2
        hh[3, :] = (hc[1, :] + hc[3, :]) / 2
        hh[0, :] = (hc[0, :] + hc[2, :]) / 2
        hh[2, :] = (hc[0, :] + hc[2, :]) / 2

        area = hh[3, :] * hh[2, :]

        ss = np.zeros(nza, dtype="float")
        ii = np.zeros(nza, dtype="int")
        jj = np.zeros(nza, dtype="int")

        nn = 0
        for n in range(n2d):
            no = nn
            for m in range(4):
                if ee_pos[m, n] != n and self._mask_n[ee_pos[m, n]] != 0:
                    nn += 1
                    # print(f"nn: {nn} m: {m} n: {n}")
                    ss[nn] = (hc[m, n] / hh[m, n]) / area[n]
                    ii[nn] = n
                    jj[nn] = ee_pos[m, n]

            ii[no] = n
            jj[no] = n
            ss[no] = -np.sum(ss[no : nn + 1])
            nn += 1

        self._ss = ss
        self._ii = ii
        self._jj = jj
        self._area = area

        # Create a mask where both _ii and _jj are not 0
        mask_sp = np.logical_and(self._mask_n[ii], self._mask_n[jj])

        self._ss = self._ss[mask_sp]
        self._ii = self._ii[mask_sp]
        self._jj = self._jj[mask_sp]

        self._ss, self._ii, self._jj, self._vcycle_weight = _symmetrize_stencil(
            self._ss, self._ii, self._jj, self._area
        )



    def _compute(
        self,
        n: int,
        k: float | np.ndarray,
        data: np.ndarray,
        gamma: float = 2.0,
        x0: np.ndarray | None = None,
        maxiter: int = 150_000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Solve the implicit filter system, stage by stage.

        For n >= 3 the operator ``I + gamma*S**n`` is factorised into stages of
        order <= 2 (Danilov et al. 2024); see
        :func:`implicit_filter.utils.utils.filter_stages`. Each stage solves
        ``(I + c1*S + c2*S**2) x = b`` in perturbation form, and the output of
        one stage is the input of the next. For n = 1, 2 there is a single
        stage and this reduces to the original single-system formula.

        ``filter_stages`` only factorises n = 1, 2, 3, 4; unlike the
        pre-staging implementation, arbitrary n is no longer supported (this
        matches :meth:`TriangularFilter._compute`'s constraint, taken on for
        the same reason: it's the staged factorisation that keeps CG's
        condition number bounded at higher order).
        """
        # jax_enable_x64 is on, so Smat1_diag below (built from a bare
        # jnp.zeros, no dtype) is float64 regardless of data's dtype; solving
        # in float64 throughout avoids a float32 apply_A silently meeting a
        # float64 preconditioner mid-CG, which jax's while_loop rejects as a
        # carry dtype mismatch. Cast back to the caller's dtype on return.
        input_dtype = np.asarray(data).dtype

        k_arg = k  # pre-broadcast value; the V-cycle path needs a scalar k
        if isinstance(k, (float, int, np.number)):
            k = np.ones(self._n2d) * k

        scaling_vector = -1.0 / np.square(k)
        data_smat1 = self._ss * scaling_vector[self._jj]

        diag_mask = self._ii == self._jj
        Smat1_diag = jnp.zeros(self._n2d).at[self._ii[diag_mask]].add(data_smat1[diag_mask])

        stages = filter_stages(n, gamma)
        use_vcycle = self.get_preconditioner() == "vcycle"

        tts = jnp.array(data, dtype=jnp.float64)
        for (c1, c2) in stages:
            # c1/c2 are bound as defaults so the closure captures this stage's
            # coefficients rather than the loop variables' final values.
            def apply_A(x, c1=c1, c2=c2):
                Sx = jnp.zeros_like(x).at[self._ii].add(data_smat1 * x[self._jj])
                y = x + c1 * Sx
                if c2 != 0.0:
                    y = y + c2 * jnp.zeros_like(x).at[self._ii].add(data_smat1 * Sx[self._jj])
                return y

            # Approximate diagonal of this stage. diag(S**2) is approximated by
            # diag(S)**2, as in the original implementation.
            approx_diag_Smat = 1.0 + c1 * Smat1_diag + c2 * (Smat1_diag ** 2)

            def precond(x, d=approx_diag_Smat):
                return x / d

            ttw = tts - apply_A(tts)  # Work with perturbations
            # A warm start only has a meaningful shape for a single-stage
            # solve; with several stages the caller's x0 refers to the final
            # field, not to this stage's intermediate solution.
            x0_pert = (None if (x0 is None or len(stages) > 1)
                       else (jnp.array(x0, dtype=jnp.float64) - tts))

            # Per-stage preconditioner choice: mirrors TriangularFilter's
            # rationale -- a linear stage (c2 == 0) stays cheap for Jacobi,
            # while a quadratic stage is where the V-cycle pays off.
            if use_vcycle and c2 != 0.0:
                from implicit_filter.utils._vcycle import (
                    solve_with_vcycle, validate_scalar_k)

                # The lat-lon stencil is assembled negative-semidefinite (the
                # solve scales by -1/k^2), so the PSD-convention stencil is -S.
                # The symmetrizing weight is area for a spherical grid
                # (zonal spacing scaled by cos(latitude), breaking the
                # separable-tensor-product assumption below) but area^2 for a
                # stretched, unscaled Cartesian tensor grid (hh separable in
                # x and y) -- each is exact for one family and badly wrong
                # for the other, so prepare() picks whichever nearly
                # symmetrizes D @ S for this grid and caches it as
                # self._vcycle_weight (see _symmetrize_stencil). prepare()
                # also symmetrizes D @ S once at assembly time under that
                # weight: it is only exact to float64 roundoff otherwise
                # (each edge is computed independently from both endpoints),
                # and that residual gets amplified by orders of magnitude
                # when S is squared for a quadratic filter stage. (Curvilinear
                # grids such as NEMO's ORCA are not tensor-product; their
                # stencil is structurally asymmetric under any diagonal
                # weight and the V-cycle setup rejects them.)
                sol = solve_with_vcycle(
                    ss=-np.asarray(self._ss), ii=self._ii, jj=self._jj,
                    area=self._vcycle_weight,
                    n_size=int(self._n2d), n=n, stage=(c1, c2),
                    k=validate_scalar_k(k_arg), apply_A=apply_A,
                    b_pert=ttw, x0_pert=x0_pert, tol=tol, maxiter=maxiter,
                    options=self.preconditioner_options,
                    cache=self.vcycle_cache, tag="latlon")
            else:
                M = None if self.get_preconditioner() == "none" else precond
                sol, code = cg(apply_A, ttw, x0=x0_pert, tol=tol, maxiter=maxiter, M=M)
                if code is not None and code != 0:
                    raise SolverNotConvergedError(
                        "Solver has not converged without metric terms",
                        [f"output code with code: {code}"],
                    )

            tts = sol + tts  # add the perturbation back; input of the next stage

        return np.array(tts, dtype=input_dtype)

    def _compute_batch(
        self,
        n: int,
        k: float | np.ndarray,
        data: np.ndarray,
        gamma: float = 2.0,
        x0: np.ndarray | None = None,
        maxiter: int = 150_000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Batched counterpart to :meth:`_compute`: solves the same per-stage
        system for a whole leading batch of right-hand sides (e.g. time
        steps or depth levels) at once via ``jax.vmap`` over the matrix-free
        stencil, instead of looping :meth:`compute`/:meth:`compute_velocity`
        one field at a time.

        Parameters
        ----------
        data : np.ndarray, shape (T, n2d)
            Batch of right-hand sides.
        x0 : np.ndarray, shape (T, n2d), optional
            Batch of warm-start fields; see :meth:`_compute`'s caveat that a
            warm start is only used for a single-stage solve (n = 1 or 2).

        With the V-cycle preconditioner selected (:meth:`set_preconditioner`),
        each stage's setup (mesh + stage + k, independent of the batch) is
        built once via
        :func:`implicit_filter.utils._vcycle.prepare_vcycle_preconditioner`
        and then applied to the whole batch via
        :func:`implicit_filter.utils._vcycle.solve_with_vcycle_batch`; unlike
        the single-field V-cycle path, this forgoes the automatic
        retry-at-tighter-tolerance on non-convergence (it needs a concrete
        residual norm, which isn't available under ``jax.vmap``).
        """
        # See _compute's note: Smat1_diag is float64 regardless of data's
        # dtype (jax_enable_x64 is on), so the solve runs in float64
        # throughout to avoid a carry dtype mismatch under jax.vmap; the
        # result is cast back to the caller's dtype on return.
        input_dtype = np.asarray(data).dtype

        k_arg = k  # pre-broadcast value; the V-cycle path needs a scalar k
        if isinstance(k, (float, int, np.number)):
            k = np.ones(self._n2d) * k

        scaling_vector = -1.0 / np.square(k)
        data_smat1 = self._ss * scaling_vector[self._jj]

        diag_mask = self._ii == self._jj
        Smat1_diag = jnp.zeros(self._n2d).at[self._ii[diag_mask]].add(data_smat1[diag_mask])

        stages = filter_stages(n, gamma)
        use_vcycle = self.get_preconditioner() == "vcycle"

        tts = jnp.array(data, dtype=jnp.float64)
        x0_j = None if x0 is None else jnp.array(x0, dtype=jnp.float64)

        for (c1, c2) in stages:
            def apply_A(x, c1=c1, c2=c2):
                Sx = jnp.zeros_like(x).at[self._ii].add(data_smat1 * x[self._jj])
                y = x + c1 * Sx
                if c2 != 0.0:
                    y = y + c2 * jnp.zeros_like(x).at[self._ii].add(data_smat1 * Sx[self._jj])
                return y

            approx_diag_Smat = 1.0 + c1 * Smat1_diag + c2 * (Smat1_diag ** 2)

            def precond(x, d=approx_diag_Smat):
                return x / d

            ttw = tts - jax.vmap(apply_A)(tts)
            x0_pert = (None if (x0_j is None or len(stages) > 1)
                       else (x0_j - tts))

            if use_vcycle and c2 != 0.0:
                from implicit_filter.utils._vcycle import (
                    solve_with_vcycle_batch, validate_scalar_k)

                # See _compute's note: the lat-lon stencil is assembled
                # negative-semidefinite, so the PSD-convention stencil is -S,
                # and the symmetrizing weight is self._vcycle_weight (area or
                # area^2, whichever this grid needs -- see prepare()).
                sol = solve_with_vcycle_batch(
                    ss=-np.asarray(self._ss), ii=self._ii, jj=self._jj,
                    area=self._vcycle_weight,
                    n_size=int(self._n2d), n=n, stage=(c1, c2),
                    k=validate_scalar_k(k_arg),
                    apply_A=apply_A, b_pert_batch=ttw, x0_pert_batch=x0_pert,
                    tol=tol, maxiter=maxiter, options=self.preconditioner_options,
                    cache=self.vcycle_cache, tag="latlon")
            else:
                M = None if self.get_preconditioner() == "none" else precond
                solve_one = lambda b, x0b: cg(apply_A, b, x0=x0b, tol=tol,
                                               maxiter=maxiter, M=M)[0]
                if x0_pert is None:
                    sol = jax.vmap(lambda b: solve_one(b, None))(ttw)
                else:
                    sol = jax.vmap(solve_one)(ttw, x0_pert)

            tts = sol + tts

        return np.array(tts, dtype=input_dtype)

    def compute(self, n: int, k: float | np.ndarray, data: np.ndarray, x0: np.ndarray | None = None, *,
                highpass: bool = True, gamma: float | None = None) -> np.ndarray:
        """
        Apply filter to scalar field(s) on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of a single field (n2d = nx*ny), not the
            batch dimension.
        data : np.ndarray
            Scalar field values on grid. Either a single field, shape (nx, ny), or a leading
            batch of fields (e.g. time steps or depth levels), shape (T, nx, ny) — the batch
            case is dispatched to :meth:`_compute_batch` via `jax.vmap` instead of looping
            :meth:`_compute` T times.
        x0 : np.ndarray | None
            Initial guess for the solver. Same batching convention as `data`.
        highpass : bool, optional
            Whether to use high-pass filtering. Sets gamma (2.0 vs 0.5) unless
            gamma is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma value. If None, derived from highpass. Default None.

        Returns
        -------
        np.ndarray
            Filtered scalar field(s), same shape as `data` — (nx, ny) or (T, nx, ny).

        Raises
        ------
        ValueError
            If filter order n < 1, or `data` isn't shaped (nx, ny) or (T, nx, ny).
        """
        if n < 1:
            raise ValueError("Filter order must be positive")
        g = get_gamma(highpass, gamma)

        data = np.asarray(data)
        grid_shape = (self._nx, self._ny)
        if data.shape == grid_shape:
            x0_flat = np.reshape(x0, self._n2d) if x0 is not None else None
            return np.reshape(
                self._compute(n, k, np.reshape(data, self._n2d), gamma=g, x0=x0_flat), grid_shape
            )
        elif data.ndim == 3 and data.shape[1:] == grid_shape:
            warnings.warn(
                "Batch filtering: dispatching to _compute_batch via jax.vmap",
                stacklevel=2)
            batch = data.shape[0]
            x0_flat = np.reshape(x0, (batch, self._n2d)) if x0 is not None else None
            return np.reshape(
                self._compute_batch(n, k, np.reshape(data, (batch, self._n2d)), gamma=g, x0=x0_flat),
                (batch, *grid_shape),
            )
        else:
            raise ValueError(
                f"data must be shaped {grid_shape} or (T, *{grid_shape}), got {data.shape}"
            )

    def compute_velocity(
        self, n: int, k: float | np.ndarray, ux: np.ndarray, vy: np.ndarray, ux0: np.ndarray | None = None,
        vy0: np.ndarray | None = None, *, highpass: bool = True, gamma: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to velocity components on lat-lon grid.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data
        ux : np.ndarray
            Eastward velocity component (shape: (nx, ny)).
        vy : np.ndarray
            Northward velocity component (shape: (nx, ny)).
        ux0 : np.ndarray | None
            Initial guess for ux.
        vy0 : np.ndarray | None
            Initial guess for vy.
        highpass : bool, optional
            Whether to use high-pass filtering. Sets gamma (2.0 vs 0.5) unless
            gamma is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma value. If None, derived from highpass. Default None.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered velocity components (ux_filt, vy_filt) each with shape (nx, ny).

        Raises
        ------
        ValueError
            If filter order n < 1.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")
        g = get_gamma(highpass, gamma)

        ux0_flat = np.reshape(ux0, self._n2d) if ux0 is not None else None
        vy0_flat = np.reshape(vy0, self._n2d) if vy0 is not None else None

        return (
            np.reshape(
                self._compute(n, k, np.reshape(ux, self._n2d), gamma=g, x0=ux0_flat), (self._nx, self._ny)
            ),
            np.reshape(
                self._compute(n, k, np.reshape(vy, self._n2d), gamma=g, x0=vy0_flat), (self._nx, self._ny)
            ),
        )

    def compute_spectra_scalar(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute power spectra for scalar field at specified wavelengths.

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        data : np.ndarray
            Scalar field values on grid (shape: (nx, ny)).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.

        Returns
        -------
        np.ndarray
            Power spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total variance
            [1:] : Variance at each wavelength k
        """
        nr = len(k)
        tt = np.reshape(data, self._n2d)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask: np.ndarray = np.zeros(tt.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]

        spectra[0] = np.sum(selected_area * (np.square(tt))[not_mask]) / np.sum(
            selected_area
        )

        x0 = None
        for i in range(nr):
            ttu = self._compute(n, k[i], tt, x0=x0)
            x0 = ttu
            
            ttu_diff = ttu - tt
            ttu_diff[mask] = 0.0
            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu_diff))[not_mask]
            ) / np.sum(selected_area)

        return spectra

    def compute_spectra_velocity(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute power spectra for velocity field at specified wavelengths.

        If one want's to use spatialy varying filter scale, k should be
        list of numpy arrays with size mathing the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        ux : np.ndarray
            Eastward velocity component (shape: (nx, ny)).
        vy : np.ndarray
            Northward velocity component (shape: (nx, ny)).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.

        Returns
        -------
        np.ndarray
            Kinetic energy spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total kinetic energy
            [1:] : Kinetic energy at each wavelength k
        """

        nr = len(k)
        unod = np.reshape(ux, self._n2d)
        vnod = np.reshape(vy, self._n2d)

        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(unod.shape, dtype=bool)

        not_mask = ~mask
        selected_area = self._area[not_mask]
        spectra[0] = np.sum(
            selected_area * (np.square(unod) + np.square(vnod))[not_mask]
        ) / np.sum(selected_area)

        ux0, vy0 = None, None
        for i in range(nr):
            ttu = self._compute(n, k[i], unod, x0=ux0)
            ttv = self._compute(n, k[i], vnod, x0=vy0)
            ux0, vy0 = ttu, ttv

            ttu_diff = ttu - unod
            ttv_diff = ttv - vnod

            ttu_diff[mask] = 0.0
            ttv_diff[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu_diff) + np.square(ttv_diff))[not_mask]
            ) / np.sum(selected_area)

        return spectra
