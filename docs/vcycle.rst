V-cycle preconditioner
======================

When you need it
----------------

The default Jacobi-preconditioned CG struggles when the filter is *stiff* —
the biharmonic operator (``n=2``) at large filter-scale-to-resolution
ratios. The operator ``A = I + 2 (S/k²)ⁿ`` then has eigenvalues spanning
roughly :math:`10^9`–:math:`10^{10}`; one-level preconditioning cannot
compress that spectrum, so CG needs thousands of iterations or fails to
converge at all (the breakdown regions documented in Danilov et al. 2024).

The V-cycle preconditioner replaces the diagonal scaling with one geometric
multigrid cycle per CG iteration: cheap Chebyshev smoothing handles the
rough error components on each level of a mesh hierarchy, and an *exact*
dense solve on a ~1000-unknown coarsest level supplies the global coupling
a one-level method fundamentally lacks. Iteration counts drop from
thousands (or divergence) to tens, essentially independent of mesh size.

Filter orders above 2 are supported as well and are *pure* failure region
for Jacobi-CG: on the validation mesh Jacobi-CG converges for none of the
tested ``n = 3, 4, 5`` configurations (200,000-iteration cap) while the
V-cycle solves them in 68–212 iterations (tested through ``n = 5``,
identically on CPU and GPU). Iteration counts grow mildly with the order,
and so does the operator's condition number — ``(L/dx)^{2n}`` — which
bounds the attainable accuracy in float64; keep ``n`` and the
scale-to-resolution ratio jointly sane.

Usage
-----

Requires the optional setup-time dependencies::

    pip install "implicit_filter[vcycle]"

.. code-block:: python

    flter = FesomFilter.load_from_file("filter_cache.npz")
    flter.set_backend("gpu")              # optional; CPU works identically
    flter.set_preconditioner("vcycle")
    filtered = flter.compute(2, 2 * math.pi / distance, data)

``set_preconditioner`` accepts ``'jacobi'`` (the default), ``'none'``
(plain CG) and ``'vcycle'``. The choice is runtime state, like the
backend: it applies to all subsequent ``compute*`` calls on the instance
and is not stored by :meth:`save_to_file`.

How it works
------------

The stencil ``S = D⁻¹K`` (``D`` = lumped areas, ``K`` symmetric) is not
symmetric, but ``Â = D·A`` is symmetric positive definite. When the
V-cycle is active the CG solves the symmetrized system ``(D·A)x = D·b``,
preconditioned by one V-cycle on ``Â``:

* **Hierarchy** (once per mesh, independent of ``k`` and ``n``):
  smoothed-aggregation coarsening of ``K = D·S`` via pyamg, seeded and
  bit-reproducible, coarsening ~6–10× per level down to ≤ 1000 unknowns.
* **Per (k, n) setup** (sub-second at 10⁵ nodes, cached on the filter):
  Galerkin coarse operators ``Â_{l+1} = PᵀÂ_lP`` with a hard symmetry
  check, Chebyshev spectral bounds by seeded power iteration (×1.1
  safety), dense Cholesky factorization of the coarsest level.
* **Apply** (inside CG): degree-3 Chebyshev pre/post-smoothing per level
  plus the exact coarse solve — matvec-only JAX code that runs unchanged
  on CPU and GPU. One V-cycle iteration costs several operator
  applications (measured ≈5–7 on the benchmark meshes; the exact factor
  is recorded per mesh in the benchmark report), so at 10–1000× fewer
  iterations the net win is large.
* **Convergence gate**: CG stops on the D-weighted residual, so the
  unweighted residual ``‖b − Ax‖/‖b‖`` is verified after the solve (with
  one bounded retry at a tighter tolerance) and
  :class:`SolverNotConvergedError` is raised if the requested tolerance
  was not reached — unlike the default path, where JAX's CG reports no
  convergence status.

Options
-------

Advanced knobs (defaults are evidence-backed; none needed per-mesh
tuning):

===============  =========  =====================================================
option           default    meaning
===============  =========  =====================================================
``degree``       3          Chebyshev degree per pre/post smooth
``alpha``        4.0        smoothing interval ``[λmax/α, λmax]``
``n_cycles``     1          V-cycles per CG iteration
``max_levels``   6          hierarchy depth
``max_coarse``   1000       direct-solve threshold (dense Cholesky)
``seed``         42         hierarchy + power-iteration seed
``lam_safety``   1.1        safety factor on the smoothing bound
``strength``     symmetric  pyamg strength-of-connection measure
===============  =========  =====================================================

.. code-block:: python

    flter.set_preconditioner("vcycle", degree=4, max_coarse=500)

Limitations
-----------

* **Scalar filter scales only.** Spatially varying ``k`` breaks the
  symmetry of ``D·A`` (it would need a diagonal similarity transform);
  requesting the V-cycle with a varying ``k`` raises ``ValueError``.
* **Metric terms unsupported.** The coupled ``full=True`` velocity system
  has block structure whose symmetrization is future work;
  ``NotImplementedError`` is raised.
* **The mesh must yield a symmetric operator.** Setup verifies every
  level and refuses structurally asymmetric stencils with a clear error.
  In practice: triangular meshes (FESOM, ICON) and all tensor-product
  lat-lon grids work — including arbitrarily *stretched* axes, which are
  symmetrized exactly by an internal ``area²`` weighting. Curvilinear
  grids (``NemoFilter``'s ORCA ``mesh_mask`` grids: measured 0.6 relative
  asymmetry, grid-wide, irreparable by any diagonal weighting — the
  symmetrized operator is even indefinite) are rejected and stay on
  Jacobi; a symmetric reassembly of that stencil is possible follow-up
  work. Caches saved by older package versions in float32 produce a
  harmless storage-roundoff warning.
* The preconditioner choice is not persisted by ``save_to_file``.

Benchmarks
----------

Measured before/after comparisons (CORE2 126k-node FESOM mesh and the
7.4M-node ICON grid, CPU and GPU, Jacobi vs V-cycle, with provenance) are
committed at ``docs/benchmarks/vcycle_comparison.md``. Headline: stiff
configurations where Jacobi-CG does not converge at all are solved by the
V-cycle in tens of iterations; at the production tolerance the GPU solve
of a 126k-node biharmonic filter takes a few hundredths of a second.

References
----------

* S. Danilov, C. Stepanov et al. (2024) — implicit filtering breakdown
  analysis for biharmonic filters at large scale ratios.
* K. Nowak et al. (2025), *Implicit filtering on unstructured meshes*,
  arXiv:2404.07398.
* Project design notes:
  ``docs/superpowers/specs/2026-07-28-vcycle-preconditioner-design.md``.
