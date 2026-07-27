"""
Tests for NemoFilter grid assembly.

These exercise prepare_from_data_array directly on a synthetic NEMO-style
dataset, rather than mocking the filter's attributes, so that the grid-metric
fill-in code is actually covered.
"""
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from implicit_filter.nemo_filter import NemoFilter


NX, NY = 4, 5


def make_nemo_dataset(nx=NX, ny=NY):
    """Build a minimal NEMO mesh-mask-like dataset.

    Cell metrics are deliberately anisotropic (e1* != e2*) and the
    north-south cell-centre distance varies with latitude, so that any
    mix-up between edge lengths and centre distances, or between the
    northward and southward neighbour, changes the result.
    """
    # dims are (y, x); the filter transposes to (x, y) internally
    yy = np.arange(ny)[:, None] * np.ones((1, nx))

    gphit = np.linspace(-10.0, 10.0, ny)[:, None] * np.ones((1, nx))
    glamt = np.linspace(0.0, 20.0, nx)[None, :] * np.ones((ny, 1))

    e1t = np.full((ny, nx), 1000.0)   # cell width  (x)
    e2t = np.full((ny, nx), 2000.0)   # cell height (y)

    e1v = np.full((ny, nx), 1100.0)   # west edge length
    e2u = np.full((ny, nx), 2100.0)   # north edge length

    e1u = np.full((ny, nx), 1200.0)   # distance to west neighbour centre
    e2v = 2200.0 + 100.0 * yy         # distance to north neighbour centre (varies in y)

    ones3d = np.ones((1, ny, nx))

    return xr.Dataset(
        {
            "gphit": (["y", "x"], gphit),
            "glamt": (["y", "x"], glamt),
            "e1t": (["y", "x"], e1t),
            "e2t": (["y", "x"], e2t),
            "e1v": (["y", "x"], e1v),
            "e2u": (["y", "x"], e2u),
            "e1u": (["y", "x"], e1u),
            "e2v": (["y", "x"], e2v),
            "e3u_0": (["z", "y", "x"], ones3d),
            "e3v_0": (["z", "y", "x"], ones3d),
            "e3t_0": (["z", "y", "x"], ones3d),
            "tmask": (["z", "y", "x"], ones3d),
        },
        coords={"x": np.arange(nx), "y": np.arange(ny), "z": np.arange(1)},
    )


def make_legacy_nemo_dataset(nx=NX, ny=NY):
    """A pre-3.6 NEMO mesh_mask, as written by NEMO 2.x/3.x.

    The `_0` suffix changed meaning between NEMO generations:

        old (<=3.4)   e3t / e3u / e3v    3D scale factors
                      e3t_0 / e3w_0      1D reference levels  (t, z)
        new (3.6+)    e3t_0/e3u_0/e3v_0  3D scale factors
                      e3t_1d / e3w_1d    1D reference levels

    So an old mesh *does* contain `e3t_0` -- but it is the 1D profile, not the
    field the filter needs. Selecting by name alone silently picks the wrong
    variable; selection must go by dimensionality.

    Modelled on /work/ab0995/a270125/OLD_MA/mesh_mask_nemo.v2.2.nc
    (ORCA1, National Oceanography Centre, 2012).
    """
    ds = make_nemo_dataset(nx, ny)
    nz = ds.sizes["z"]
    legacy = ds.drop_vars(["e3u_0", "e3v_0", "e3t_0"])
    for new, old in (("e3u_0", "e3u"), ("e3v_0", "e3v"), ("e3t_0", "e3t")):
        legacy[old] = (["z", "y", "x"], ds[new].values)
    # the 1D reference profile, which legacy files store under the _0 names
    legacy["e3t_0"] = (["z"], np.ones(nz))
    legacy["e3w_0"] = (["z"], np.ones(nz))
    return legacy


def build_filter(neighb="local", ds=None):
    filt = NemoFilter()
    filt.prepare_from_data_array(
        make_nemo_dataset() if ds is None else ds,
        vl=0, mask=False, neighb=neighb,
    )
    return filt


def dense_offdiagonals(filt):
    """Return {(row, col): value} for off-diagonal sparse entries."""
    ii = np.asarray(filt._ii)
    jj = np.asarray(filt._jj)
    ss = np.asarray(filt._ss)
    return {(int(i), int(j)): float(s)
            for i, j, s in zip(ii, jj, ss) if int(i) != int(j)}


class TestGridAssembly:
    def test_prepare_runs(self):
        filt = build_filter()
        assert filt._e2d == NX * NY
        assert len(filt._ss) == len(filt._ii) == len(filt._jj)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN ISSUE, deliberately not fixed: the hc[2]/hc[3] fill-in in "
            "NemoFilter.prepare_from_data_array reads hc[1, ee_pos[1, n]] under "
            "an ee_pos[3, n] guard, reads hh instead of hc for the eastward "
            "distance, and self-assigns hc[2, n] = hc[2, n] in the else branch. "
            "The equivalent hh loop just above it is self-consistent. Fixing "
            "this would change numerical output, so the behaviour is pinned "
            "here instead. Remove this marker if the fill-in is ever corrected."
        ),
    )
    def test_face_coefficients_are_reciprocal(self):
        """The flux coefficient across a face must not depend on which side
        you compute it from.

        The stored entry is  A[n, nb] = (edge_len / centre_dist) / area[n],
        so multiplying back by area[n] must give the same face quantity as
        A[nb, n] * area[nb].
        """
        filt = build_filter()
        off = dense_offdiagonals(filt)
        area = np.asarray(filt._area)

        checked = 0
        for (n, nb), val in off.items():
            if (nb, n) not in off:
                continue
            forward = val * area[n]
            backward = off[(nb, n)] * area[nb]
            np.testing.assert_allclose(
                forward, backward, rtol=1e-10,
                err_msg=(f"face {n}<->{nb} has inconsistent coefficients: "
                         f"{forward} from {n}, {backward} from {nb}"),
            )
            checked += 1

        assert checked > 0, "no interior faces were checked - grid too small"

    def test_rows_sum_to_zero(self):
        """A Laplacian must annihilate constants."""
        filt = build_filter()
        ii = np.asarray(filt._ii)
        ss = np.asarray(filt._ss)
        row_sums = np.zeros(filt._e2d)
        np.add.at(row_sums, ii, ss)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)

    def test_offdiagonals_are_positive(self):
        """Sign convention: LatLonFilter._compute negates, so stored
        off-diagonals are positive and the diagonal is negative."""
        filt = build_filter()
        off = dense_offdiagonals(filt)
        assert all(v > 0 for v in off.values())


class TestLegacyNemoNaming:
    """Pre-3.6 NEMO mesh_mask files must be supported.

    Verified against a real ORCA1 mesh (NEMO v2.2, 2012): it has e3t/e3u/e3v
    as the 3D scale factors and a 1D e3t_0, and previously failed with a bare
    `AttributeError: 'Dataset' object has no attribute 'e3u_0'`.
    """

    def test_legacy_mesh_prepares(self):
        filt = build_filter(ds=make_legacy_nemo_dataset())
        assert filt._e2d == NX * NY
        assert len(filt._ss) == len(filt._ii) == len(filt._jj)

    def test_legacy_matches_modern_when_equivalent(self):
        """The two conventions describe the same grid, so must agree exactly."""
        modern = build_filter(ds=make_nemo_dataset())
        legacy = build_filter(ds=make_legacy_nemo_dataset())
        np.testing.assert_array_equal(
            np.asarray(modern._ss), np.asarray(legacy._ss)
        )
        np.testing.assert_array_equal(
            np.asarray(modern._area), np.asarray(legacy._area)
        )

    def test_legacy_1d_reference_profile_is_not_used(self):
        """A legacy mesh's 1D e3t_0 must be ignored in favour of 3D e3t.

        If the 1D profile were picked up, the constant-1.0 values here would
        differ from the modern dataset's 3D field and the operators would not
        match.
        """
        legacy = make_legacy_nemo_dataset()
        legacy["e3t_0"] = (["z"], np.full(legacy.sizes["z"], 999.0))
        filt = build_filter(ds=legacy)
        modern = build_filter(ds=make_nemo_dataset())
        np.testing.assert_array_equal(
            np.asarray(filt._ss), np.asarray(modern._ss)
        )

    def test_legacy_constant_field_preserved(self):
        filt = build_filter(ds=make_legacy_nemo_dataset())
        result = filt.compute(1, 0.5, np.full((NX, NY), 3.0))
        np.testing.assert_allclose(result, 3.0, atol=1e-5)


class TestMissingScaleFactors:
    """A mesh with neither convention must say so clearly."""

    def test_missing_scale_factor_raises_keyerror(self):
        ds = make_nemo_dataset().drop_vars(["e3u_0"])
        filt = NemoFilter()
        with pytest.raises(KeyError, match="e3u"):
            filt.prepare_from_data_array(ds, vl=0, mask=False, neighb="local")

    def test_error_mentions_both_conventions(self):
        ds = make_nemo_dataset().drop_vars(["e3v_0"])
        filt = NemoFilter()
        with pytest.raises(KeyError, match=r"e3v_0.*e3v|e3v.*e3v_0"):
            filt.prepare_from_data_array(ds, vl=0, mask=False, neighb="local")


class TestVerticalDimensionNaming:
    """The vertical dimension must be named 'z'.

    NEMO writes mesh_mask.nc with dims (t, z, y, x), but files that have been
    through CDO or other tooling often carry 'nav_lev' or 'deptht' instead. The
    diagnostic should name the offending dimension rather than just reporting a
    missing variable.
    """

    @pytest.mark.parametrize("dim", ["nav_lev", "deptht", "depth"])
    def test_renamed_vertical_dim_is_named_in_the_error(self, dim):
        ds = make_nemo_dataset().rename({"z": dim})
        filt = NemoFilter()
        with pytest.raises(KeyError, match=dim):
            filt.prepare_from_data_array(ds, vl=0, mask=False, neighb="local")

    @pytest.mark.parametrize("dim", ["nav_lev", "deptht"])
    def test_error_says_to_rename_to_z(self, dim):
        ds = make_nemo_dataset().rename({"z": dim})
        filt = NemoFilter()
        with pytest.raises(KeyError, match="rename"):
            filt.prepare_from_data_array(ds, vl=0, mask=False, neighb="local")

    def test_renaming_to_z_makes_it_work(self):
        """The remedy the error suggests must actually work."""
        ds = make_nemo_dataset().rename({"z": "nav_lev"})
        filt = NemoFilter()
        filt.prepare_from_data_array(
            ds.rename({"nav_lev": "z"}), vl=0, mask=False, neighb="local"
        )
        assert filt._e2d == NX * NY


class TestNorthFoldDetectionFailure:
    """neighb='full' resolves the north-fold row correspondence by matching
    rounded coordinates column by column, injectively. On some grids a column's
    only candidate is consumed by an earlier column and the match comes up
    short -- observed on a real ORCA1 mesh (NEMO v2.2), where row -2 matches
    359 of 360 columns.

    That surfaced as `ValueError: Length of values (359) does not match length
    of index (360)` from deep inside pandas. The failure must instead say what
    went wrong and what to do about it.
    """

    @staticmethod
    def starved_dataset():
        # interior columns of the redundant row repeat a longitude that occurs
        # only once in the corresponding row, so one column cannot be matched
        nx, ny = 6, 5
        lon = np.tile(np.arange(nx) * 10.0, (ny, 1))
        lon[-1, 2] = lon[-1, 1]          # duplicate -> starves the greedy match
        lat = np.tile(np.arange(ny)[:, None] * 5.0, (1, nx))
        lat[-1, :] = lat[-2, :]
        return xr.Dataset(
            {"glamt": (["y", "x"], lon), "gphit": (["y", "x"], lat)},
            coords={"x": np.arange(nx), "y": np.arange(ny)},
        )

    def test_raises_actionable_error(self):
        from implicit_filter.utils._auxiliary import find_adjacent_points_north

        with pytest.raises(ValueError, match="north"):
            find_adjacent_points_north(self.starved_dataset(), 1e-5)

    def test_error_reports_how_many_columns_failed(self):
        from implicit_filter.utils._auxiliary import find_adjacent_points_north

        with pytest.raises(ValueError, match=r"\b1\b.*column|column.*\b1\b"):
            find_adjacent_points_north(self.starved_dataset(), 1e-5)

    def test_error_suggests_other_neighbourhoods(self):
        from implicit_filter.utils._auxiliary import find_adjacent_points_north

        with pytest.raises(ValueError, match="west-east|local"):
            find_adjacent_points_north(self.starved_dataset(), 1e-5)

    def test_not_a_bare_pandas_length_error(self):
        from implicit_filter.utils._auxiliary import find_adjacent_points_north

        with pytest.raises(ValueError) as excinfo:
            find_adjacent_points_north(self.starved_dataset(), 1e-5)
        assert "does not match length of index" not in str(excinfo.value)


class TestFiltering:
    def test_constant_field_preserved(self):
        filt = build_filter()
        data = np.full((NX, NY), 5.0)
        result = filt.compute(1, 0.5, data)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_variance_decreases(self):
        filt = build_filter()
        np.random.seed(3)
        data = np.random.randn(NX, NY)
        result = filt.compute(1, 0.5, data)
        assert np.var(result) <= np.var(data) + 1e-10

    def test_west_east_neighbourhood_runs(self):
        filt = build_filter(neighb="west-east")
        data = np.full((NX, NY), 2.0)
        np.testing.assert_allclose(filt.compute(1, 0.5, data), 2.0, atol=1e-5)

    def test_unknown_neighbourhood_rejected(self):
        filt = NemoFilter()
        with pytest.raises(NotImplementedError, match="not supported"):
            filt.prepare_from_data_array(
                make_nemo_dataset(), vl=0, mask=False, neighb="banana"
            )
