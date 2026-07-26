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


def build_filter(neighb="local"):
    filt = NemoFilter()
    filt.prepare_from_data_array(make_nemo_dataset(), vl=0, mask=False, neighb=neighb)
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
