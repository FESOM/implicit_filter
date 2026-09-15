# Changelog

## Unreleased

### Packaging

- **The `cuda11` extra is gone.** It pointed at `jax[cuda11_pip]`, which JAX
  no longer provides, so `pip install "implicit_filter[cuda11]"` silently
  installed a CPU-only build. JAX has dropped CUDA 11 support; use CUDA 12.
- **The `apple` extra now installs `jax-metal`.** It pointed at `jax[apple]`,
  which does not exist either and likewise installed CPU-only.
- **New `rocm` and `tpu` extras**, forwarding to `jax[rocm]` and `jax[tpu]`, so
  AMD GPUs and Google TPUs are installable the same way as CUDA. The ROCm
  plugin targets ROCm 6.x and expects a matching ROCm installation on the
  machine.
- **scikit-learn is no longer a dependency.** It was pulled into every install
  (57 MB) to fit the single straight line that resolves the NEMO north-fold row
  correspondence; that fit is now an explicit numpy least-squares solve, which
  agrees with the scikit-learn fit to within floating-point rounding -- far
  below the integer rounding the helper applies to it -- on the column indices
  this code path produces. pandas is still required, and still only by that
  helper. Where `LinearRegression` used to raise on an empty set of surviving
  columns, `np.linalg.lstsq` would have returned a degenerate all-zero line, so
  the helper now refuses explicitly when fewer than two columns survive its
  outlier filter -- a grid that scattered was never fittable.
- The cupy-era extras `gpu`, `gpu_c11`, `gpu_c12` and `amgx` were removed when
  the package moved to JAX. If you still install one of those, pip only warns
  (`WARNING: implicit-filter ... does not provide the extra 'gpu'`) and installs
  the base package, giving you a CPU-only build: use `cuda12` for Nvidia GPUs,
  `apple` for Apple Silicon, and `vcycle` for what `amgx` used to provide.

### Compatibility fixes against the previous release

- **A failed solve raises again.** The default solver's status check was
  inherited from scipy's CG and could never fire under JAX, whose `cg` returns
  no status: a diverged solve, or one whose input contained NaN or inf, was
  returned as if it were a filtered field. Every default-path solve now
  verifies the true relative residual, retries once at a tenth of the
  tolerance (as the V-cycle path already did) and otherwise raises
  `SolverNotConvergedError`. Solves that already converged are returned
  unchanged, bit for bit. Filtering data that contains NaN now raises instead
  of silently returning it; replace non-finite values before filtering and use
  `mask=` to exclude those points.
- **`set_backend` validates its argument again.** Anything other than `'cpu'`
  or `'gpu'` raises `NotImplementedError("Backend <name> is not supported.")`
  as it did before the JAX migration, instead of being silently taken to mean
  GPU. Case and surrounding whitespace are now ignored, so `'GPU'` is accepted.
- **`gpu=True` selects the GPU again.** The deprecated `gpu=` argument of the
  `prepare*` methods used to select the GPU backend; since the JAX migration it
  only warned, silently moving those calls onto the CPU. It now forwards to
  `set_backend('gpu')` and still warns. `gpu=False`, the default, remains a
  no-op and will not reset a backend already chosen.
- **Importing the package switches JAX to 64-bit mode process-wide.** This has
  been true since the JAX migration and is now documented: results are
  `float64` rather than JAX's default `float32`, and other JAX code in the same
  interpreter is affected. See the precision note in the readme.

### Faster `prepare` on large meshes

The mesh topology and geometry loops that `TriangularFilter.prepare` runs
before assembling the operator (`neighboring_triangles`, `neighbouring_nodes`
and `areas`) are now vectorised NumPy (`implicit_filter.utils._fast_mesh`).
Results are bit-identical to the pure-Python versions, which stay in
`_auxiliary` as the tested references: the same neighbour ordering, the same
floating-point operation order per triangle, and node areas summed in NumPy's
own order. Meshes with non-float64 coordinates fall back to the reference
loops on NumPy 1.x, or when `cyclic_length` is a NumPy scalar or the mask is an
integer array, so results stay bit-identical there too. On the N320 reduced
Gaussian grid (542 080 nodes) the three stages went from 15 s to 1.7 s and
`prepare_from_grid("N320")` from about 33 s to about 20 s; the JAX operator
assembly is unchanged.

### Reduced Gaussian grids (ECMWF N and O grids)

New `ReducedGaussianFilter` for data on ECMWF reduced Gaussian grids —
classical `N` grids such as N320 (native ERA5, AIFS input) and octahedral `O`
grids such as O96 (Anemoi ERA5, AIFS training) — or any other set of points
covering the sphere. `prepare_from_grid("N320")` builds the grid from ECMWF's
definitions (classical tables regenerated from the ecCodes GRIB samples;
octahedral grids from `4·j + 16` points per row); `prepare_from_points`,
`prepare_from_data_array` and `prepare_from_file(..., engine="cfgrib")` use
the coordinates stored with the data. Data is the 1-D GRIB `values` array
(north to south, west to east). An optional per-point `mask` excludes e.g.
land points.

The grid is Delaunay-triangulated on the sphere (convex hull of the unit
vectors) and filtered with the existing nodal operator using a new
`meshtype="s"`: each triangle's geometry is evaluated in the tangent plane at
its centroid (gnomonic projection), which stays well-behaved at the poles and
across the date line. In the lon/lat projection of `meshtype="r"` the triangles
that close a polar cap have zero area, which poisons the operator with NaN and
makes `compute` return the input unchanged. Measured against the analytic
transfer function for spherical harmonics on O96 the new geometry is accurate
to 1e-4 – 2e-3 (the expected discretisation error). The `'m'` and `'r'`
branches are untouched; `prepare()` now rejects unknown `meshtype` values and
refuses `meshtype="s"` with `cartesian=True` or `filter_elements=True`.

New public helpers `gaussian_latitudes(N)`, `reduced_gaussian_grid(name)` and
`spherical_triangulation(lat, lon)`; the latter refuses point sets that are
regional or leave a hole (the origin must lie strictly inside their convex
hull, and no triangle may span a gap much wider than the local point spacing),
and `check_coverage=False` — also accepted by `prepare_from_points`,
`prepare_from_data_array` and `prepare_from_file` — skips the heuristic part
of that check for complete grids it wrongly rejects, such as one with a dense
polar ring. `scipy` (already required by JAX) is now listed explicitly in the
requirements. Example on public ERA5 data:
`examples/reduced_gaussian_grid_example.ipynb`.

### V-cycle preconditioner (opt-in)

New `set_preconditioner(name, **options)` / `get_preconditioner()` on every
filter, mirroring `set_backend`. Choices: `'jacobi'` (default — numerically
identical to before), `'none'` (plain CG), and `'vcycle'` — a geometric
multigrid V-cycle (smoothed-aggregation hierarchy, Chebyshev(3) smoothing,
exact coarse solve) applied to the symmetrized SPD system `(D·A)x = D·b`.
It eliminates the Jacobi-CG convergence failures for stiff biharmonic
configurations, with identical results on CPU and GPU (parity-tested).
Setup needs the new optional extra `implicit_filter[vcycle]` (pyamg +
scipy); the apply phase is pure JAX. Details, tuning knobs and measured
before/after benchmarks: `docs/vcycle.rst`.

Notes:

- The V-cycle path verifies the true (unweighted) residual after the solve
  and raises `SolverNotConvergedError` if the tolerance was not met. The
  default path now does the same (see the compatibility fixes above), so a
  failed solve raises on either path.
- Supported systems: triangular nodes and elements (FESOM, ICON) and all
  tensor-product lat-lon grids, including stretched axes (symmetrized
  exactly by an internal `area²` weighting); spatially varying `k` and
  the metric-terms (`full=True`) system raise clear errors; curvilinear
  stencils (NEMO/FOCI ORCA `mesh_mask` grids, 0.6 relative asymmetry
  under any diagonal weighting) are refused at setup.
- `set_backend("gpu")` now selects the concrete `cuda` platform when the
  split CUDA plugin (`jax[cuda12]`) is installed — with the plugin, JAX's
  `"gpu"` alias also probes a ROCm stub whose failure made GPU selection
  crash. CPU behaviour and `get_backend()` round-trips are unchanged.
- The `gpu=` argument on every `prepare*` method is deprecated. Passing a
  truthy value emits a `DeprecationWarning` and forwards to
  `set_backend('gpu')`; the default (`False`) stays silent and leaves the
  selected backend alone, and the argument will be removed in a future
  release.

### Backward compatibility

Audited against the pre-review baseline by running identical code on both
revisions and diffing the results.

**Nothing was removed.** No package-level export, class, method or function
disappeared. Every signature change is a parameter *appended at the end with a
default* (`elem_weights`, `filter_elements`, `on`), so existing positional and
keyword calls bind exactly as before. `test_backward_compatibility.py` pins
this.

**Cache files interoperate in both directions.** A cache written by the previous
version loads here, and a cache written here loads under the previous version,
both producing identical filtered output.

Three behaviour changes are worth knowing about:

| change | who is affected | impact |
| --- | --- | --- |
| `get_backend()` returns `'gpu'` instead of `'gpu,cpu'` | code comparing the return value to the literal `"gpu,cpu"` | now round-trips through `set_backend()`, which the old value did not |
| ICON `mask=True` on a grid whose mask is all zeros now raises | the 6 such grids in the public pool | previously returned the input **completely unfiltered** (all-zero operator), silently |
| filtered values shift on spherical/non-uniform meshes | see the float32 note below | ~1e-9 – 1e-7 relative |

Everything else — filtering on nodes and elements, velocity, spectra,
`full=True`, `LatLonFilter`, the conversion helpers, error types for invalid
input — was verified byte-for-byte identical.

`pandas` was briefly moved to an optional `[nemo]` extra and has been
**restored to the base requirements**: `neighb='full'` is `NemoFilter`'s
default, so a plain `pip install implicit_filter` must keep working on that
path. It is still imported lazily and still produces a clear message naming the
extra if absent. (`scikit-learn` was restored alongside it at the time; it has
since been dropped from the package entirely -- see Packaging under Unreleased
above.)

The trimming of `requirements.txt` does mean packages the old pin set installed
transitively (`matplotlib`, `requests`, `Bottleneck`, `numexpr`, and build tools
like `pip`/`setuptools`/`wheel`) are no longer pulled in. The package never
imported them; only a user script that relied on `implicit_filter` to supply
them would notice.

### ⚠️ Changes that affect numerical output

Two fixes in this release change computed results. Everything else in it was
verified bit-identical to the previous version on Cartesian and spherical
meshes, for nodal and element filtering, velocity, spectra, `full=True`, and
`LatLonFilter`.

#### ICON land-sea mask now actually masks land

`IconFilter.prepare_from_file` / `prepare_from_data_array` with `mask=True`
previously produced a filter **bit-identical to `mask=False`** — no land was
ever excluded. The branch flipped the sign of `cell_sea_land_mask`, applied a
recoding, then overwrote the result with a fresh read, and finished with
`astype(bool)`, which maps every nonzero ICON code (`-2, -1, +1, +2`) to `True`.

Ocean is now `cell_sea_land_mask < 0`.

**Convention verified against the ICON grid pool**, not assumed. All 50 grids in
`/pool/data/ICON/grids/public` (providers `edzw` and `mpim`) that carry the
variable — spanning **2016-12-13 to 2025-10-08 and 15 distinct grid-generator
revisions** — declare the identical `long_name`:

> `sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell`

No grid contains any value outside `{-2,-1,0,1,2}`. Two populated variants occur
and both are handled: global (`_G`) grids use all four codes, while ocean (`_O`)
grids omit inner land and use `{-2,-1,+1}`. Checked against the real files, the
resulting ocean fraction is ~69% for global grids (consistent with Earth's ~71%
ocean) and 94–96% for ocean-only grids.

#### ICON grids with an unpopulated mask are now rejected

Six grids in the same pool — including `icon_grid_0005_R02B04_G.nc`,
`icon_grid_0030_R02B03_G.nc`, `icon_grid_0037_R02B11_G.nc` and the Torus test
grid — contain `cell_sea_land_mask` filled entirely with zeros. Their land-sea
information lives in a sibling file (`icon_grid_0030_R02B03_Glsm.nc`,
`icon_grid_0023_R02B07_G_slm.nc`).

Under `ocean == code < 0` such a grid would mark **every cell as land**, giving
zero areas everywhere and a silently degenerate filter. (The previous code was
equally wrong here, for a different reason.) `mask=True` on a grid with no sea
cells now raises `ValueError` naming the sibling-file convention and the
`mask=False` / explicit-array alternatives.

**Impact.** Measured on the ICON R2B04 grid `icon_grid_0043_R02B04_G.nc`
(20480 cells, 30.7% land), filtering an SST-like field with land cells zeroed:

| filter scale | max change | rms change | ocean nodes >1% of span |
| --- | --- | --- | --- |
| n=1, L=300 km | 2.68 °C (11.8% of span) | 0.33 °C | 9.9% |
| n=1, L=500 km | 5.21 °C (23.0%) | 0.65 °C | 15.5% |
| n=1, L=1000 km | 9.08 °C (41.0%) | 1.20 °C | 27.4% |

Integrated element area was previously inflated by 44%, which also affected the
normalisation in `compute_spectra_scalar` / `compute_spectra_velocity`.

**Who is affected.** Only `IconFilter` users who passed `mask=True`. Passing a
precomputed `np.ndarray` mask, or `mask=False`, was and remains unaffected.
Results produced with `mask=True` should be regarded as unmasked.

#### Filter operator assembled in double precision

`make_smooth` accumulated the operator into a `float32` buffer while all inputs
were `float64` and `jax_enable_x64` was enabled, truncating the operator to
~24 bits. It now accumulates in the inputs' dtype.

**Impact.** Bit-identical on uniform Cartesian meshes — which is why no existing
test changed. On spherical or non-uniform meshes roughly `1e-9`–`1e-7` relative
change in filtered fields (measured up to `1.6e-4` on a 0.5° spherical mesh at
high latitude). This is well below the CG solver's default `tol=1e-6` for
oscillatory fields, but the solver contributes exactly zero error to the
large-scale/DC component, where the truncation was previously the entire error.

Also restores the exact constant-null-space property (operator row sums
`6e-7` → `2e-15`) and removes a JAX `FutureWarning` that is scheduled to become
a hard error in `prepare()`.

### NEMO mesh compatibility

Verified against real NEMO mesh-mask files of two generations. Output for
already-supported meshes is bit-identical; the changes only affect meshes that
previously failed.

| mesh | `local` | `west-east` | `full` |
| --- | --- | --- | --- |
| ORCA05, NEMO ≥3.6 (722×511) | ✅ unchanged | ✅ unchanged | ✅ unchanged |
| ORCA1, NEMO v2.2 (362×292) | ✅ **now works** | ✅ **now works** | ⚠️ clear error |

#### Pre-3.6 NEMO mesh files are now supported

The meaning of the `_0` suffix changed between NEMO generations:

| | NEMO ≤3.4 | NEMO ≥3.6 |
| --- | --- | --- |
| 3D scale factors | `e3t`, `e3u`, `e3v` | `e3t_0`, `e3u_0`, `e3v_0` |
| 1D reference levels | `e3t_0`, `e3w_0` | `e3t_1d`, `e3w_1d` |

`NemoFilter` hardcoded the ≥3.6 names, so an older mesh failed with a bare
`AttributeError: 'Dataset' object has no attribute 'e3u_0'`. Both conventions
are now accepted.

Selection is by **dimensionality, not name**: a legacy mesh does contain
`e3t_0`, but as the 1D reference profile. A name-based fallback would silently
pick a 1D array of the wrong length instead of the 3D field, so a candidate is
accepted only if it carries `(z, y, x)`.

#### Clearer diagnostics for unsupported meshes

- **North-fold detection.** `neighb='full'` matches the redundant northern row
  against its partner column by column, greedily and injectively. On grids where
  a column's only candidate is already taken the match comes up short — on the
  real ORCA1 mesh, 359 of 360 columns match — which surfaced as an opaque
  `ValueError: Length of values (359) does not match length of index (360)` from
  inside pandas. It now reports how many columns failed and points to
  `neighb='west-east'` / `'local'`. The matching itself is unchanged, as
  altering it would change results on grids where it currently succeeds.
- **Vertical dimension naming.** The code requires the vertical dimension to be
  named `z` (as NEMO writes it), but files processed through CDO often carry
  `nav_lev` or `deptht`. That now names the offending dimension and gives the
  `ds.rename({...: 'z'})` remedy, instead of reporting a missing variable.

### Fixed

- `LatLonFilter.prepare` with a land mask produced misaligned sparse arrays
  (`_ii` was not filtered alongside `_ss`/`_jj`), so any masked lat-lon filter
  raised `ValueError` on first use. The unmasked path is unchanged.
- `save_to_file` / `load_from_file` round-trip failed for the default
  `filter_elements=False` configuration: unset attributes were pickled as
  `None`, which `np.load(..., allow_pickle=False)` then refused. Unset
  attributes are now omitted on save and restored as `None` on load.
- `neighbouring_nodes` had a fixed 20-entry scratch buffer and raised
  `IndexError` on meshes with higher node valence.
- The Earth radius was hardcoded three times, with the element branch using
  6371 km while the nodal geometry used 6400 km. All sites now share
  `_auxiliary.R_EARTH = 6400.0`. Verified numerically inert: the element-path
  value never reached the assembled operator.
- `get_backend()` returned JAX's internal priority string (`"gpu,cpu"`), which
  `set_backend()` did not accept. It now returns `"cpu"` or `"gpu"`.
- `np.bool` (removed in NumPy 1.24–1.26) replaced with `np.bool_`.
- `TriangularFilter(mapping)` and `LatLonFilter(mapping)` raised `TypeError`
  when constructed from a positional state mapping.

### Added

- `elem_weights` on `prepare()`: `'equilateral'` (default, unchanged behaviour)
  or `'geometric'`, which uses the per-edge weights computed from the actual
  mesh instead of the fixed `sqrt(3)/area` equilateral coefficient. The two
  agree on an equilateral mesh and diverge as cells become anisotropic.
- `filter_elements` and `elem_weights` are now reachable through
  `FesomFilter` and `IconFilter` `prepare_from_file` / `prepare_from_data_array`;
  previously element filtering required the low-level `TriangularFilter.prepare`.
- `on='nodes'|'elements'` on `compute` / `compute_velocity`, to disambiguate
  placement on meshes with as many elements as nodes.

### Packaging

- `requirements.txt` reduced from a 31-line frozen environment dump (including
  `pip`, `wheel`, `setuptools`, `munkres`, `kiwisolver`, …) to the four direct
  runtime dependencies.
- `pandas` and `scikit-learn` moved to an optional `[nemo]` extra; they are
  imported lazily and only by the NEMO north-fold helper, which now raises a
  message naming the extra when they are absent. (Both were restored to the
  base requirements later in this same release; `scikit-learn` has since been
  dropped from the package entirely -- see Packaging under Unreleased above.)
- Added a `[test]` extra (`pytest`, `scipy`) — `scipy` was previously an
  undeclared test dependency that resolved only because JAX happens to require
  it. CI now installs it explicitly.
- Corrected the invalid trove classifier `Development Status :: 5 - Production`
  to `5 - Production/Stable`.

### Known issues

Two defects are documented by strict-`xfail` tests rather than fixed, because
correcting them would change numerical output:

- `NemoFilter.prepare_from_data_array` fills the eastward/southward cell-centre
  distances inconsistently: it reads `hc[1, ee_pos[1, n]]` under an
  `ee_pos[3, n]` guard, reads `hh` instead of `hc` for the eastward distance,
  and self-assigns `hc[2, n] = hc[2, n]`. The equivalent `hh` loop immediately
  above is self-consistent. See `test_nemo_filter.py`.
- `find_adjacent_points_north` compares only longitude when choosing which row
  the redundant northern row folds onto, and builds `ilat_redundant` /
  `ilat_corresponds` from `ilon`. On grids where longitude does not vary along
  `y` it cannot distinguish the rows. `test_auxiliary.py::test_find_adjacent_points_north`
  fails for this reason.
