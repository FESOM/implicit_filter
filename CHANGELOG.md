# Changelog

## Unreleased

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
  message naming the extra when they are absent.
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
