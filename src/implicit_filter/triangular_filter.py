from math import pi
from typing import Tuple, Iterable

import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax

jax.config.update("jax_platforms", "cpu")

from implicit_filter.utils._auxiliary import (
    R_EARTH,
    neighboring_triangles,
    neighbouring_nodes,
    areas,
    find_and_sort_edges_and_triangles,
    calculate_triangle_centers,
)
from implicit_filter.utils._jax_elem_function import (
    vectorized_orient_edges,
    vectorized_calculate_dimensional_quantities,
    fast_calculate_laplacian_weights,
    fast_build_smoothing_and_metric,
    fast_assemble_from_intermediate,
)
from implicit_filter.utils._jax_function import (
    make_smooth,
    make_smat,
    make_smat_full,
    transform_mask_to_nodes,
)
from implicit_filter.utils.utils import (
    SolverNotConvergedError,
    transform_attribute,
    warn_unused_gpu_argument,
    filter_stages,
    get_gamma,
)
from jax.scipy.sparse.linalg import cg
from implicit_filter.filter import Filter


class TriangularFilter(Filter):
    """
    A class for filtering data using JAX for generic triangular meshes.
    Extends the base Filter class.

    Attributes:
    -----------
    _elem_area : Optional[jnp.ndarray]
        Area of each element in the mesh.
    _area : Optional[jnp.ndarray]
        Area of each node's neighborhood in the mesh.
    _ne_pos : Optional[jnp.ndarray]
        Connectivity matrix representing neighboring elements for each node.
    _ne_num : Optional[jnp.ndarray]
        Number of neighboring elements for each node.
    _dx : Optional[jnp.ndarray]
        X-component of the derivative of P1 basis functions.
    _dy : Optional[jnp.ndarray]
        Y-component of the derivative of P1 basis functions.
    _ss : Optional[jnp.ndarray]
        Non-zero entries of the sparse matrix.
    _ii : Optional[jnp.ndarray]
        Row indices of non-zero entries.
    _jj : Optional[jnp.ndarray]
        Column indices of non-zero entries.
    _n2d : int
        Total number of nodes in the mesh.
    _full : bool
        Flag indicating whether to use the full matrix.
    _mask_n: Optional[jnp.ndarray]
        Mask of valid elements in the mesh.
        For example it can land mask.


    Methods:
    --------
    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute filtered velocity components (u, v) using implicit filtering.
    Compute(n: int, k: float, data: np.ndarray) -> np.ndarray:
        Compute filtered data using implicit filtering.
    Prepare(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
            cartesian: bool, cyclic_length: float, full: bool = False):
        Prepare the filter for a specific mesh.

    """

    def __init__(self, *initial_data, **kwargs):
        """
        Initialize the Triangular filter instance.

        Parameters:
        -----------
        *initial_data : positional arguments
            Positional arguments passed to the base class constructor.
        **kwargs : keyword arguments
            Keyword arguments passed to the base class constructor.
        """

        super().__init__(*initial_data, **kwargs)
        # Transform to JAX array
        jx = lambda ar: jnp.array(ar)
        bl = lambda ar: bool(ar)
        it = lambda ar: int(ar)
        st = lambda ar: str(ar)

        transform_attribute(self, "_elem_area", jx, None)
        transform_attribute(self, "_area", jx, None)
        transform_attribute(self, "_ne_pos", jx, None)
        transform_attribute(self, "_en_pos", jx, None)
        transform_attribute(self, "_ne_num", jx, None)
        transform_attribute(self, "_dx", jx, None)
        transform_attribute(self, "_dy", jx, None)

        transform_attribute(self, "_ss", jx, None)
        transform_attribute(self, "_ii", jx, None)
        transform_attribute(self, "_jj", jx, None)
        transform_attribute(self, "_ss_e", jx, None)
        transform_attribute(self, "_ii_e", jx, None)
        transform_attribute(self, "_jj_e", jx, None)
        transform_attribute(self, "_mask_n", jx, None)

        transform_attribute(self, "_n2d", it, 0)
        transform_attribute(self, "_e2d", it, 0)
        transform_attribute(self, "_full", bl, False)

    def _resolve_target(self, length: int, on: str | None) -> bool:
        """
        Decide whether data of the given length lives on elements or nodes.

        Parameters
        ----------
        length : int
            Length of the data array.
        on : {'nodes', 'elements', None}
            Explicit placement. ``None`` infers it from the length, which is
            ambiguous when the mesh has as many elements as nodes.

        Returns
        -------
        bool
            True if the data is on elements.
        """
        if on is None:
            return length == self._e2d
        if on == "elements":
            if length != self._e2d:
                raise ValueError(
                    f"data length {length} does not match the element count "
                    f"{self._e2d} implied by on='elements'"
                )
            return True
        if on == "nodes":
            if length != self._n2d:
                raise ValueError(
                    f"data length {length} does not match the node count "
                    f"{self._n2d} implied by on='nodes'"
                )
            return False
        raise ValueError(
            f"on={on!r} is not valid; expected 'nodes', 'elements' or None"
        )

    def _compute(self, n, kl, ttu, gamma=2.0, x0=None, tol=1e-6, maxiter=150000,
                 is_elem=None) -> np.ndarray:
        """
        Solve the implicit filter system, stage by stage.

        For n >= 3 the operator ``I + gamma*S**n`` is factorised into stages of
        order <= 2 (Danilov et al. 2024); see
        :func:`implicit_filter.utils.utils.filter_stages`. Each stage solves
        ``(I + c1*S + c2*S**2) x = b`` in perturbation form, and the output of
        one stage is the input of the next. For n = 1, 2 there is a single
        stage and this reduces to the original single-system formula.
        """
        kl_arg = kl  # pre-broadcast value; the V-cycle path needs a scalar k
        if is_elem is None:
            is_elem = (len(ttu) == self._e2d)
        if is_elem and self._ss_e is None:
            raise ValueError("Filter was not prepared with filter_elements=True")

        n_size = self._e2d if is_elem else self._n2d
        ss = self._ss_e if is_elem else self._ss
        ii = self._ii_e if is_elem else self._ii
        jj = self._jj_e if is_elem else self._jj

        if isinstance(kl, (float, int, np.number)):
            kl = np.ones(ttu.shape) * kl

        scaling_vector = 1.0 / np.square(kl)
        data = ss * scaling_vector[jj]

        diag_mask = ii == jj
        Smat1_diag = jnp.zeros(n_size).at[ii[diag_mask]].add(data[diag_mask])

        stages = filter_stages(n, gamma)
        use_vcycle = self.get_preconditioner() == "vcycle"


        tts = jnp.array(ttu)
        for (c1, c2) in stages:
            # c1/c2 are bound as defaults so the closure captures this stage's
            # coefficients rather than the loop variables' final values.
            def apply_A(x, c1=c1, c2=c2):
                Sx = jnp.zeros_like(x).at[ii].add(data * x[jj])
                y = x + c1 * Sx
                if c2 != 0.0:
                    y = y + c2 * jnp.zeros_like(x).at[ii].add(data * Sx[jj])
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
                       else (jnp.array(x0) - tts))

            # Per-stage preconditioner choice: a linear stage (c2 == 0) stays
            # cheap for Jacobi even at large filter scales, while a quadratic
            # stage is biharmonic-conditioned and is where the V-cycle pays
            # off. Measured on a 15k-node mesh at L = 1000 km: 154 Jacobi
            # iterations for the linear stage of n = 3 against 5443 for the
            # quadratic one.
            if use_vcycle and c2 != 0.0:
                from implicit_filter.utils._vcycle import (
                    solve_with_vcycle, validate_scalar_k)

                sol = solve_with_vcycle(
                    ss=ss, ii=ii, jj=jj,
                    area=self._elem_area if is_elem else self._area,
                    n_size=n_size, n=n, stage=(c1, c2),
                    k=validate_scalar_k(kl_arg),
                    apply_A=apply_A, b_pert=ttw, x0_pert=x0_pert,
                    tol=tol, maxiter=maxiter, options=self.preconditioner_options,
                    cache=self.vcycle_cache, tag="elem" if is_elem else "node")
            else:
                M = None if self.get_preconditioner() == "none" else precond
                sol, code = cg(apply_A, ttw, x0=x0_pert, tol=tol,
                               maxiter=maxiter, M=M)
                if code is not None and code != 0:
                    raise SolverNotConvergedError(
                        "Solver has not converged without metric terms",
                        [f"output code with code: {code}"],
                    )

            tts = sol + tts  # add the perturbation back; input of the next stage

        return np.array(tts)

    def _compute_batch(self, n, kl, ttu, gamma=2.0, x0=None, tol=1e-6, maxiter=150000,
                        is_elem=None) -> np.ndarray:
        """
        Batched counterpart to :meth:`_compute`: solves the same per-stage
        system for a whole leading batch of right-hand sides (e.g. time
        steps or depth levels) at once via ``jax.vmap`` over the matrix-free
        stencil, instead of looping :meth:`compute`/:meth:`compute_velocity`
        one field at a time.

        Parameters
        ----------
        ttu : np.ndarray, shape (T, n_size)
            Batch of right-hand sides.
        x0 : np.ndarray, shape (T, n_size), optional
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
        kl_arg = kl  # pre-broadcast value; the V-cycle path needs a scalar k
        if is_elem is None:
            is_elem = (ttu.shape[-1] == self._e2d)
        if is_elem and self._ss_e is None:
            raise ValueError("Filter was not prepared with filter_elements=True")

        n_size = self._e2d if is_elem else self._n2d
        ss = self._ss_e if is_elem else self._ss
        ii = self._ii_e if is_elem else self._ii
        jj = self._jj_e if is_elem else self._jj

        if isinstance(kl, (float, int, np.number)):
            kl = np.ones(n_size) * kl

        scaling_vector = 1.0 / np.square(kl)
        data = ss * scaling_vector[jj]

        diag_mask = ii == jj
        Smat1_diag = jnp.zeros(n_size).at[ii[diag_mask]].add(data[diag_mask])

        stages = filter_stages(n, gamma)
        use_vcycle = self.get_preconditioner() == "vcycle"

        tts = jnp.array(ttu)
        x0_j = None if x0 is None else jnp.array(x0)

        for (c1, c2) in stages:
            def apply_A(x, c1=c1, c2=c2):
                Sx = jnp.zeros_like(x).at[ii].add(data * x[jj])
                y = x + c1 * Sx
                if c2 != 0.0:
                    y = y + c2 * jnp.zeros_like(x).at[ii].add(data * Sx[jj])
                return y

            approx_diag_Smat = 1.0 + c1 * Smat1_diag + c2 * (Smat1_diag ** 2)

            def precond(x, d=approx_diag_Smat):
                return x / d

            ttw = jax.vmap(apply_A)(tts)
            ttw = tts - ttw
            x0_pert = (None if (x0_j is None or len(stages) > 1)
                       else (x0_j - tts))

            if use_vcycle and c2 != 0.0:
                from implicit_filter.utils._vcycle import (
                    solve_with_vcycle_batch, validate_scalar_k)

                sol = solve_with_vcycle_batch(
                    ss=ss, ii=ii, jj=jj,
                    area=self._elem_area if is_elem else self._area,
                    n_size=n_size, n=n, stage=(c1, c2),
                    k=validate_scalar_k(kl_arg),
                    apply_A=apply_A, b_pert_batch=ttw, x0_pert_batch=x0_pert,
                    tol=tol, maxiter=maxiter, options=self.preconditioner_options,
                    cache=self.vcycle_cache, tag="elem" if is_elem else "node")
            else:
                M = None if self.get_preconditioner() == "none" else precond
                solve_one = lambda b, x0b: cg(apply_A, b, x0=x0b, tol=tol,
                                               maxiter=maxiter, M=M)[0]
                if x0_pert is None:
                    sol = jax.vmap(lambda b: solve_one(b, None))(ttw)
                else:
                    sol = jax.vmap(solve_one)(ttw, x0_pert)

            tts = sol + tts

        return np.array(tts)

    def _compute_full(self, n, kl, ttuv, gamma=2.0, x0=None, tol=1e-5,
                      maxiter=150000) -> np.ndarray:
        if n > 2:
            raise NotImplementedError(
                "Full metric filtering is currently only implemented for n=1 "
                "and n=2. Please use full=False for n>=3.")
        if self.get_preconditioner() == "vcycle":
            raise NotImplementedError(
                "The V-cycle preconditioner does not support the coupled "
                "metric-terms system (full=True); its block structure needs "
                "a separate symmetry analysis. Use set_preconditioner('jacobi').")
        if isinstance(kl, (float, int, np.number)):
            kl = np.ones(2 * self._n2d) * kl

        scaling_vector = 1.0 / np.square(kl)
        data = self._ss * scaling_vector[self._jj]

        def apply_A(x):
            y = x
            for _ in range(n):
                y = jnp.zeros_like(x).at[self._ii].add(data * y[self._jj])
            return x + gamma * y

        diag_mask = self._ii == self._jj
        Smat1_diag = jnp.zeros(2 * self._n2d).at[self._ii[diag_mask]].add(data[diag_mask])
        approx_diag_Smat = 1.0 + gamma * (Smat1_diag ** n)

        def precond(x):
            return x / approx_diag_Smat

        ttuv = jnp.array(ttuv)
        ttw = ttuv - apply_A(ttuv)  # Work with perturbations
        x0_pert = None if x0 is None else (jnp.array(x0) - ttuv)

        tts, code = cg(apply_A, ttw, x0=x0_pert, tol=tol, maxiter=maxiter, M=precond)
        if code is not None and code != 0:
            raise SolverNotConvergedError(
                "Solver has not converged with metric terms",
                [f"output code with code: {code}"],
            )

        tts += ttuv
        return np.array(tts)

    def compute_velocity(
        self, n: int, k: float | np.ndarray, ux: np.ndarray, vy: np.ndarray,
        ux0: np.ndarray | None = None, vy0: np.ndarray | None = None,
        on: str | None = None, *, highpass: bool = True,
        gamma: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to velocity components on triangular mesh.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data
        ux : np.ndarray
            Eastward velocity component.
        vy : np.ndarray
            Northward velocity component.
        ux0 : np.ndarray | None
            Initial guess for ux.
        vy0 : np.ndarray | None
            Initial guess for vy.
        on : {'nodes', 'elements'}, optional
            Where the data lives. Inferred from length by default; pass it
            explicitly on a mesh with as many elements as nodes.
        highpass : bool, optional
            Whether to use high-pass filtering. Sets gamma (2.0 vs 0.5) unless
            gamma is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma value. If None, derived from highpass. Default None.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered velocity components (ux_filt, vy_filt).

        Raises
        ------
        ValueError
            If filter order n < 1, or if ``on`` is unrecognised or disagrees
            with the length of the input.
        SolverNotConvergedError
            If linear solver fails to converge.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")
        g = get_gamma(highpass, gamma)

        uxn = ux
        vyn = vy

        is_elem = self._resolve_target(len(uxn), on)
        if is_elem and self._full:
            raise ValueError("Coupled full metric filtering not supported for elements. Please use full=False or switch to nodal filtering.")

        if self._full and not is_elem:
            uv0 = np.concatenate((ux0, vy0)) if ux0 is not None and vy0 is not None else None
            ttuv = self._compute_full(n, k, np.concatenate((uxn, vyn)), gamma=g, x0=uv0)
            return ttuv[0 : self._n2d], ttuv[self._n2d : 2 * self._n2d]
        else:
            ttu = self._compute(n, k, uxn, gamma=g, x0=ux0, is_elem=is_elem)
            ttv = self._compute(n, k, vyn, gamma=g, x0=vy0, is_elem=is_elem)
            return ttu, ttv

    def compute(self, n: int, k: float | np.ndarray, data: np.ndarray,
                x0: np.ndarray | None = None, on: str | None = None, *,
                highpass: bool = True, gamma: float | None = None) -> np.ndarray:
        """
        Apply filter to scalar field on triangular mesh.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : float | np.ndarray
            Filter wavelength in spatial units.
            Float can be passed to be applied for entire mesh or array with scales for each node.
            Size of the array must match the size of the input data
        data : np.ndarray
            Scalar field values.
        x0 : np.ndarray | None
            Initial guess for the solver.
        on : {'nodes', 'elements'}, optional
            Where the data lives. By default this is inferred from the length
            of ``data``, which is ambiguous on a mesh with as many elements as
            nodes; pass it explicitly to remove the ambiguity.
        highpass : bool, optional
            Whether to use high-pass filtering. Sets gamma (2.0 vs 0.5) unless
            gamma is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma value. If None, derived from highpass. Default None.

        Returns
        -------
        np.ndarray
            Filtered scalar field.

        Raises
        ------
        ValueError
            If filter order n < 1, or if ``on`` is unrecognised or disagrees
            with the length of ``data``.
        """
        if n < 1:
            raise ValueError("Filter order must be positive")
        g = get_gamma(highpass, gamma)

        is_elem = self._resolve_target(len(data), on)
        if is_elem and self._full:
            raise ValueError("Coupled full metric filtering not supported for elements. Please use full=False or switch to nodal filtering.")

        return np.array(
            self._compute_full(n, k, data, gamma=g, x0=x0)
            if (self._full and not is_elem)
            else self._compute(n, k, data, gamma=g, x0=x0, is_elem=is_elem)
        )

    def prepare(
        self,
        n2d: int,
        e2d: int,
        tri: np.ndarray,
        xcoord: np.ndarray,
        ycoord: np.ndarray,
        meshtype: str = "r",
        cartesian: bool = True,
        cyclic_length: float = 360.0 * pi / 180.0,
        full: bool = False,
        mask: np.ndarray | None = None,
        gpu: bool = False,
        filter_elements: bool = False,
        elem_weights: str = "equilateral",
    ):
        """
        Prepare filter for a specific triangular mesh.

        The ``gpu`` argument is deprecated (it never had an effect); use
        :meth:`set_backend` instead.

        Computes mesh topology, geometric properties, and assembles the filter
        operator matrix. Must be called before any filtering operations.

        Parameters
        ----------
        n2d : int
            Number of nodes in the mesh.
        e2d : int
            Number of elements in the mesh.
        tri : np.ndarray
            Element connectivity matrix (e2d x 3) of node indices.
        xcoord : np.ndarray
            X-coordinates of mesh nodes (degrees).
        ycoord : np.ndarray
            Y-coordinates of mesh nodes (degrees).
        meshtype : str, optional
            Mesh type coordinate unit: 'm' for metric, 'r' for radial(degrees).
        cartesian : bool, optional
            True for Cartesian coordinates, False for spherical.
        cyclic_length : float, optional
            Cyclic domain length in radians (default: 2π).
        full : bool, optional
            True to include metric terms in operator (default: False).
        mask : np.ndarray, optional
            Element mask where True indicates ocean (default: all ocean).
        gpu : bool, optional
            Deprecated and without effect; select the backend with
            :meth:`set_backend` instead.
        filter_elements : bool, optional
            True to assemble filter operators for elements in addition to nodes (default: False).
        elem_weights : {'equilateral', 'geometric'}, optional
            Weighting used for the element (triangle) Laplacian. Ignored unless
            ``filter_elements=True``.

            - ``'equilateral'`` (default): fixed ``-sqrt(3)/elem_area``
              off-diagonal. This is the finite-volume coefficient
              ``edge_length/centroid_distance`` evaluated for an equilateral
              triangle, so it depends only on cell area and ignores cell shape.
              This is the historical behaviour and is kept as the default so
              that existing results remain reproducible.
            - ``'geometric'``: uses the per-edge weights actually computed from
              the mesh geometry. Identical to ``'equilateral'`` on an
              equilateral mesh; more accurate on anisotropic or graded meshes.

        Raises
        ------
        ValueError
            If ``elem_weights`` is not one of the supported schemes.

        Notes
        -----
        Coordinates are expected in degrees while cyclic_length is in radians.
        The mask is converted to nodal representation where True indicates land.
        """
        warn_unused_gpu_argument(gpu)
        if elem_weights not in ("equilateral", "geometric"):
            raise ValueError(
                f"Unknown elem_weights {elem_weights!r}; "
                "expected 'equilateral' or 'geometric'"
            )
        # NOTE: xcoord & ycoord are in degrees, but cyclic_length is in radians
        self._n2d = n2d
        self._e2d = e2d
        self._full = full

        if mask is None:
            mask = np.ones(e2d, dtype=np.bool_)
        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(
            n2d,
            e2d,
            tri,
            xcoord,
            ycoord,
            ne_num,
            ne_pos,
            meshtype,
            cartesian,
            cyclic_length,
            mask,
        )

        self._elem_area = jnp.array(elem_area)
        self._dx = jnp.array(dx)
        self._dy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        self._en_pos = jnp.array(tri.T)  # element positions in terms of nodes
        self._ne_num = jnp.array(ne_num)
        self._ne_pos = jnp.array(ne_pos)
        self._area = jnp.array(area)

        self._mask_n = jnp.array(mask)

        self._mask_n = transform_mask_to_nodes(
            self._mask_n, self._ne_pos, self._ne_num, self._n2d
        )
        self._mask_n = jnp.where(self._mask_n > 0.5, 1.0, 0.0).astype(
            bool
        )  # Where there's ocean

        smooth, metric = make_smooth(
            jMt,
            self._elem_area,
            self._dx,
            self._dy,
            jnn_num,
            jnn_pos,
            jtri,
            n2d,
            e2d,
            full,
        )

        smooth = vmap(lambda n: smooth[:, n] / self._area[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / self._area[n])(jnp.arange(0, n2d)).T

        self._ss, self._ii, self._jj = (
            make_smat_full(jnn_pos, jnn_num, smooth, metric, n2d, int(jnp.sum(jnn_num)))
            if full
            else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        )

        ## Set rows (and columns!) of smooth where (node) mask is 0 (land) to 0: This enforces a Neumann BC
        #   i.e. Set _ss = 0 where mask_n[_ii] = 0 && mask_n[_jj] = 0
        # AFW
        # Create a mask where both _ii and _jj are not 0
        if full:
            mask_sp = jnp.logical_and(
                self._mask_n[self._ii % n2d], self._mask_n[self._jj % n2d]
            )
        else:
            mask_sp = jnp.logical_and(self._mask_n[self._ii], self._mask_n[self._jj])

        self._ss = self._ss[mask_sp]
        self._ii = self._ii[mask_sp]
        self._jj = self._jj[mask_sp]

        if filter_elements:
            edges, edge_tri, ed2d_in = find_and_sort_edges_and_triangles(n2d, nn_num, nn_pos, ne_num, ne_pos)
            tcenter = calculate_triangle_centers(e2d, tri, xcoord, ycoord, meshtype, cyclic_length)
            if meshtype == 'm':
                edges, edge_tri = vectorized_orient_edges(edges.shape[1], edges, edge_tri, tcenter, xcoord, ycoord)
                edge_dxdy, edge_cross_dxdy = vectorized_calculate_dimensional_quantities(edges.shape[1], ed2d_in, edges, edge_tri, tcenter, xcoord, ycoord)
            else:
                # Fall back to original for spherical meshes
                from implicit_filter.utils import _auxiliary
                from implicit_filter.utils._auxiliary import orient_edges, calculate_dimensional_quantities
                # Read at call time so the shared constant stays the single
                # source of truth (and remains patchable in tests).
                r_earth = _auxiliary.R_EARTH
                edges, edge_tri = orient_edges(edges.shape[1], edges, edge_tri, tcenter, xcoord, ycoord, meshtype, cyclic_length)
                edge_dxdy, edge_cross_dxdy = calculate_dimensional_quantities(edges.shape[1], ed2d_in, edges, edge_tri, tcenter, xcoord, ycoord, meshtype, cyclic_length, r_earth, cartesian)
            ee_pos, ee_num, weights, dxcell = fast_calculate_laplacian_weights(e2d, ed2d_in, edge_tri, edge_dxdy, edge_cross_dxdy)
            smooth_m, metric_m = fast_build_smoothing_and_metric(
                e2d, n2d, ee_num, ee_pos, elem_area, full, Mt, dxcell,
                weights=weights, scheme=elem_weights,
            )
            ss_e, ii_e, jj_e = fast_assemble_from_intermediate(e2d, ee_num, ee_pos, smooth_m)

            # Land cells and coast couplings were already zeroed during
            # assembly; drop the explicit zeros so the triplets stay compact
            # and land rows have no diagonal at all (v-cycle rejects
            # zero diagonal).
            keep = np.asarray(ss_e) != 0.0
            ss_e = np.asarray(ss_e)[keep]
            ii_e = np.asarray(ii_e)[keep]
            jj_e = np.asarray(jj_e)[keep]

            self._ss_e = jnp.array(ss_e)
            self._ii_e = jnp.array(ii_e)
            self._jj_e = jnp.array(jj_e)

    def compute_spectra_scalar(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
    ) -> np.ndarray:
        """
        Compute power spectra for scalar field at specified wavelengths.

        If one wants to use a spatially varying filter scale, k should be
        a list of numpy arrays with size matching the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        data : np.ndarray
            Scalar field values at mesh nodes (or elements).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.
        on : {'nodes', 'elements'}, optional
            Where the data lives; inferred from length by default.
        highpass : bool, optional
            If True, high-pass spectrum (difference field); if False, low-pass
            spectrum (filtered field). Also sets gamma (2.0 vs 0.5) unless gamma
            is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma. If None, derived from highpass. Default None.
        demean : bool, optional
            If True, remove the area-weighted mean (computed over the analysed
            region) before computing spectra. Default False.

        Returns
        -------
        np.ndarray
            Power spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total variance
            [1:] : Variance at each wavelength k
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(data.shape, dtype=bool)

        is_elem = self._resolve_target(len(data), on)
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        data = data.copy()  # protect the caller's array (demean writes in place)

        if demean:
            data_mean = np.sum(selected_area * data[not_mask]) / area_sum
            data[not_mask] = data[not_mask] - data_mean

        spectra[0] = np.sum(
            selected_area * (np.square(data))[not_mask]
        ) / area_sum

        for i in range(nr):
            ttu = self.compute(n, k[i], data, on=on, highpass=highpass, gamma=gamma)

            if highpass:
                ttu -= data

            ttu[mask] = 0.0
            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu))[not_mask]
            ) / area_sum

        return spectra

    def compute_spectra_scalar_many(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
        vcycle_above: float | None = None,
    ) -> np.ndarray:
        """
        Compute scalar spectra for several snapshots, wavenumber-outer.

        Same result as calling :meth:`compute_spectra_scalar` once per
        snapshot, but the loop order is reversed: the outer loop runs over
        wavenumbers, so a V-cycle setup built for one ``k`` serves every
        snapshot instead of being rebuilt for each. With the Jacobi
        preconditioner there is no setup to reuse and the order is
        irrelevant.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        data : np.ndarray
            Scalar field, shape ``(nt, N)`` — one row per snapshot.
        mask : np.ndarray, optional
            Boolean mask of length ``N``, True excludes a point. The same
            mask is applied to every snapshot.
        on : {'nodes', 'elements'}, optional
            Where the data lives; inferred from ``N`` by default.
        highpass, gamma, demean
            As in :meth:`compute_spectra_scalar`. ``demean`` removes the
            area-weighted mean of each snapshot separately.
        vcycle_above : float | None, optional
            If given, switch to the V-cycle preconditioner for filter scales
            ``2*pi/k > vcycle_above`` and to Jacobi below. The preconditioner
            in effect on entry is restored before returning. Measured on a
            15192-node mesh, the V-cycle starts to win at n >= 3 somewhere
            below L = 1000 (n=4, L=3500: 2.7 s against 66 s for Jacobi) and
            loses at small scales, so a threshold around 1000 is a reasonable
            starting point. Pass ``k`` sorted so the switch happens once.

        Returns
        -------
        np.ndarray
            Shape ``(nt, len(k) + 1)``. Column 0 is the total variance of
            each snapshot, columns 1.. the variance at each wavelength.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2-D (nt, N); got shape {data.shape}. "
                "Use compute_spectra_scalar for a single snapshot.")
        nt, N = data.shape
        nr = len(k)

        if mask is None:
            mask = np.zeros(N, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != (N,):
            raise ValueError(
                f"mask must have shape ({N},); got {mask.shape}")

        is_elem = self._resolve_target(N, on)
        placement = "elements" if is_elem else "nodes"
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        work = data.astype(np.float64, copy=True)
        if demean:
            for t in range(nt):
                m = np.sum(selected_area * work[t][not_mask]) / area_sum
                work[t][not_mask] = work[t][not_mask] - m

        spectra = np.zeros((nt, nr + 1))
        for t in range(nt):
            spectra[t, 0] = np.sum(
                selected_area * (np.square(work[t]))[not_mask]) / area_sum

        pc_entry = self.get_preconditioner()
        try:
            for i in range(nr):
                if vcycle_above is not None:
                    scale = 2.0 * np.pi / float(np.max(k[i]))
                    want = "vcycle" if scale > vcycle_above else "jacobi"
                    if self.get_preconditioner() != want:
                        self.set_preconditioner(want)
                for t in range(nt):
                    ttu = self.compute(n, k[i], work[t], on=placement,
                                       highpass=highpass, gamma=gamma)
                    if highpass:
                        ttu -= work[t]
                    ttu[mask] = 0.0
                    spectra[t, i + 1] = np.sum(
                        selected_area * (np.square(ttu))[not_mask]) / area_sum
        finally:
            if self.get_preconditioner() != pc_entry:
                self.set_preconditioner(pc_entry)

        return spectra

    def compute_spectra_velocity(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
    ) -> np.ndarray:
        """
        Compute power spectra for velocity field at specified wavelengths.

        If one wants to use a spatially varying filter scale, k should be
        a list of numpy arrays with size matching the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        ux : np.ndarray
            Eastward velocity component at mesh nodes (or elements).
        vy : np.ndarray
            Northward velocity component at mesh nodes (or elements).
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.
        on : {'nodes', 'elements'}, optional
            Where the data lives; inferred from length by default.
        highpass : bool, optional
            If True, high-pass spectrum (difference field); if False, low-pass
            spectrum (filtered field). Also sets gamma (2.0 vs 0.5) unless gamma
            is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma. If None, derived from highpass. Default None.
        demean : bool, optional
            If True, remove the area-weighted mean (computed over the analysed
            region) before computing spectra. Default False.

        Returns
        -------
        np.ndarray
            Kinetic energy spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total kinetic energy
            [1:] : Kinetic energy at each wavelength k

        Notes
        -----
        Implements spectral decomposition:
            E(k) = <(u - u_k)**2 + (v - v_k)**2>
        where (u_k, v_k) is the filtered velocity at wavelength k.
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(ux.shape, dtype=bool)

        is_elem = self._resolve_target(len(ux), on)
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        unod = ux.copy()  # protect the caller's arrays (demean writes in place)
        vnod = vy.copy()

        if demean:
            ux_mean = np.sum(selected_area * unod[not_mask]) / area_sum
            vy_mean = np.sum(selected_area * vnod[not_mask]) / area_sum
            unod[not_mask] = unod[not_mask] - ux_mean
            vnod[not_mask] = vnod[not_mask] - vy_mean

        spectra[0] = np.sum(
            selected_area * (np.square(unod) + np.square(vnod))[not_mask]
        ) / area_sum

        for i in range(nr):
            ttu, ttv = self.compute_velocity(n, k[i], unod, vnod, on=on,
                                             highpass=highpass, gamma=gamma)

            if highpass:
                ttu -= unod
                ttv -= vnod

            ttu[mask] = 0.0
            ttv[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (np.square(ttu) + np.square(ttv))[not_mask]
            ) / area_sum

        return spectra

    def compute_spectra_velocity_many(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
        vcycle_above: float | None = None,
    ) -> np.ndarray:
        """
        Compute velocity spectra for several snapshots, wavenumber-outer.

        See :meth:`compute_spectra_scalar_many` for the loop-order rationale
        and for ``vcycle_above``.

        Parameters
        ----------
        ux, vy : np.ndarray
            Velocity components, shape ``(nt, N)`` each.

        Returns
        -------
        np.ndarray
            Shape ``(nt, len(k) + 1)``; column 0 is the total kinetic energy
            of each snapshot.
        """
        ux = np.asarray(ux)
        vy = np.asarray(vy)
        if ux.ndim != 2 or ux.shape != vy.shape:
            raise ValueError(
                "ux and vy must both be 2-D (nt, N) and of equal shape; got "
                f"{ux.shape} and {vy.shape}")
        nt, N = ux.shape
        nr = len(k)

        if mask is None:
            mask = np.zeros(N, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != (N,):
            raise ValueError(f"mask must have shape ({N},); got {mask.shape}")

        is_elem = self._resolve_target(N, on)
        placement = "elements" if is_elem else "nodes"
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        uw = ux.astype(np.float64, copy=True)
        vw = vy.astype(np.float64, copy=True)
        if demean:
            for t in range(nt):
                mu = np.sum(selected_area * uw[t][not_mask]) / area_sum
                mv = np.sum(selected_area * vw[t][not_mask]) / area_sum
                uw[t][not_mask] = uw[t][not_mask] - mu
                vw[t][not_mask] = vw[t][not_mask] - mv

        spectra = np.zeros((nt, nr + 1))
        for t in range(nt):
            spectra[t, 0] = np.sum(
                selected_area * (np.square(uw[t]) + np.square(vw[t]))[not_mask]
            ) / area_sum

        pc_entry = self.get_preconditioner()
        try:
            for i in range(nr):
                if vcycle_above is not None:
                    scale = 2.0 * np.pi / float(np.max(k[i]))
                    want = "vcycle" if scale > vcycle_above else "jacobi"
                    if self.get_preconditioner() != want:
                        self.set_preconditioner(want)
                for t in range(nt):
                    ttu, ttv = self.compute_velocity(
                        n, k[i], uw[t], vw[t], on=placement,
                        highpass=highpass, gamma=gamma)
                    if highpass:
                        ttu -= uw[t]
                        ttv -= vw[t]
                    ttu[mask] = 0.0
                    ttv[mask] = 0.0
                    spectra[t, i + 1] = np.sum(
                        selected_area
                        * (np.square(ttu) + np.square(ttv))[not_mask]
                    ) / area_sum
        finally:
            if self.get_preconditioner() != pc_entry:
                self.set_preconditioner(pc_entry)

        return spectra

    def compute_spectra_cross_velocity(
        self,
        n: int,
        k: Iterable | np.ndarray,
        ux: np.ndarray,
        vy: np.ndarray,
        uxdis: np.ndarray,
        vydis: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
    ) -> np.ndarray:
        """
        Compute cross-spectra between a velocity field and a diagnostic field.

        Computes the spectral covariance
            E_cross(k) = <u_k . u_dis_k + v_k . v_dis_k>
        at each filter scale k, where (u_k, v_k) and (u_dis_k, v_dis_k) are the
        filtered (high-pass or low-pass) velocity and diagnostic fields.

        If one wants to use a spatially varying filter scale, k should be
        a list of numpy arrays with size matching the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        ux, vy : np.ndarray
            Velocity components at mesh nodes (or elements).
        uxdis, vydis : np.ndarray
            Diagnostic field components (e.g. dissipation, backscatter,
            advection) at the same locations as ux, vy.
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.
        on : {'nodes', 'elements'}, optional
            Where the data lives; inferred from length by default.
        highpass : bool, optional
            If True, high-pass cross-spectrum (difference fields); if False,
            low-pass (filtered fields). Also sets gamma (2.0 vs 0.5) unless gamma
            is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma. If None, derived from highpass. Default None.
        demean : bool, optional
            If True, remove the area-weighted mean (computed over the analysed
            region) of each field before computing cross-spectra. Default False.

        Returns
        -------
        np.ndarray
            Cross-spectral energy at wavelengths [0, k0, k1, ...]:
            [0] : Total covariance <u . u_dis + v . v_dis>
            [1:] : Covariance at each wavelength k

        Notes
        -----
        Physical interpretation: if the diagnostic field is dissipation,
        positive values indicate an energy sink at that scale; if it is a
        forcing/backscatter term, positive values indicate a source.
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(ux.shape, dtype=bool)

        is_elem = self._resolve_target(len(ux), on)
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        # protect callers' arrays (demean writes in place below)
        unod = ux.copy()
        vnod = vy.copy()
        udis = uxdis.copy()
        vdis = vydis.copy()

        if demean:
            for fld in (unod, vnod, udis, vdis):
                fld_mean = np.sum(selected_area * fld[not_mask]) / area_sum
                fld[not_mask] = fld[not_mask] - fld_mean

        spectra[0] = np.sum(
            selected_area * (unod * udis + vnod * vdis)[not_mask]
        ) / area_sum

        for i in range(nr):
            ttu, ttv = self.compute_velocity(n, k[i], unod, vnod, on=on,
                                             highpass=highpass, gamma=gamma)
            ttudis, ttvdis = self.compute_velocity(n, k[i], udis, vdis, on=on,
                                                   highpass=highpass, gamma=gamma)

            if highpass:
                ttu -= unod
                ttv -= vnod
                ttudis -= udis
                ttvdis -= vdis

            ttu[mask] = 0.0
            ttv[mask] = 0.0
            ttudis[mask] = 0.0
            ttvdis[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (ttu * ttudis + ttv * ttvdis)[not_mask]
            ) / area_sum

        return spectra

    def compute_spectra_cross_scalar(
        self,
        n: int,
        k: Iterable | np.ndarray,
        data: np.ndarray,
        data_dis: np.ndarray,
        mask: np.ndarray | None = None,
        on: str | None = None,
        *,
        highpass: bool = True,
        gamma: float | None = None,
        demean: bool = False,
    ) -> np.ndarray:
        """
        Compute cross-spectra between a scalar field and a diagnostic scalar field.

        Computes the spectral covariance
            E_cross(k) = <data_k . data_dis_k>
        at each filter scale k, where data_k and data_dis_k are the filtered
        (high-pass or low-pass) fields.

        If one wants to use a spatially varying filter scale, k should be
        a list of numpy arrays with size matching the input data.

        Parameters
        ----------
        n : int
            Filter order (must be positive).
        k : Iterable | np.ndarray
            Target wavelengths for spectral analysis.
        data : np.ndarray
            Scalar field values at mesh nodes (or elements).
        data_dis : np.ndarray
            Diagnostic scalar field at the same locations as data.
        mask : np.ndarray, optional
            Boolean mask where True excludes points from spectra computation.
        on : {'nodes', 'elements'}, optional
            Where the data lives; inferred from length by default.
        highpass : bool, optional
            If True, high-pass cross-spectrum (difference fields); if False,
            low-pass (filtered fields). Also sets gamma (2.0 vs 0.5) unless gamma
            is given explicitly. Default True.
        gamma : float | None, optional
            Explicit gamma. If None, derived from highpass. Default None.
        demean : bool, optional
            If True, remove the area-weighted mean (computed over the analysed
            region) of each field before computing cross-spectra. Default False.

        Returns
        -------
        np.ndarray
            Cross-spectral density at wavelengths [0, k0, k1, ...]:
            [0] : Total covariance <data . data_dis>
            [1:] : Covariance at each wavelength k
        """
        nr = len(k)
        spectra = np.zeros(nr + 1)
        if mask is None:
            mask = np.zeros(data.shape, dtype=bool)

        is_elem = self._resolve_target(len(data), on)
        area_arr = np.array(self._elem_area) if is_elem else np.array(self._area)

        not_mask = ~mask
        selected_area = area_arr[not_mask]
        area_sum = np.sum(selected_area)

        # protect callers' arrays (demean writes in place below)
        dnod = data.copy()
        ddis = data_dis.copy()

        if demean:
            for fld in (dnod, ddis):
                fld_mean = np.sum(selected_area * fld[not_mask]) / area_sum
                fld[not_mask] = fld[not_mask] - fld_mean

        spectra[0] = np.sum(
            selected_area * (dnod * ddis)[not_mask]
        ) / area_sum

        for i in range(nr):
            ttu = self.compute(n, k[i], dnod, on=on, highpass=highpass, gamma=gamma)
            ttudis = self.compute(n, k[i], ddis, on=on, highpass=highpass, gamma=gamma)

            if highpass:
                ttu -= dnod
                ttudis -= ddis

            ttu[mask] = 0.0
            ttudis[mask] = 0.0

            spectra[i + 1] = np.sum(
                selected_area * (ttu * ttudis)[not_mask]
            ) / area_sum

        return spectra