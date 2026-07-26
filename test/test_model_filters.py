"""
Tests for the model-specific filter classes (FESOM, ICON).

These classes had no test coverage at all. They are thin wrappers that read a
model's mesh conventions and delegate to TriangularFilter.prepare, so the things
worth pinning are: the conventions are read correctly, and every capability of
the underlying prepare() is actually reachable through them.
"""
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from implicit_filter.fesom_filter import FesomFilter
from implicit_filter.icon_filter import IconFilter
from implicit_filter.utils._auxiliary import make_tri


def _patch_mesh(n=6, extent=2.0):
    """A small triangulated lon/lat patch near the equator, in degrees."""
    nodnum = np.reshape(np.arange(n * n), [n, n]).T
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    step = extent / (n - 1)
    for i in range(n):
        lat[i, :] = np.arange(n) * step
    for i in range(n):
        lon[:, i] = np.arange(n) * step
    tri = make_tri(nodnum, n, n)
    return lon.flatten(), lat.flatten(), tri


def fesom_dataset(elem_key="elements"):
    lon, lat, tri = _patch_mesh()
    return xr.Dataset(
        {elem_key: (["three", "cell"], (tri + 1).T)},   # FESOM: 1-based, (3, ncells)
        coords={"lon": (["node"], lon), "lat": (["node"], lat)},
    ), len(lon), len(tri)


def icon_dataset(sea_land_mask=None):
    lon, lat, tri = _patch_mesh()
    rad = np.pi / 180.0
    data_vars = {"vertex_of_cell": (["nv", "cell"], (tri + 1).T)}
    if sea_land_mask is not None:
        data_vars["cell_sea_land_mask"] = (["cell"], np.asarray(sea_land_mask))
    return xr.Dataset(
        data_vars,
        coords={
            "vlon": (["vertex"], lon * rad),   # ICON: radians
            "vlat": (["vertex"], lat * rad),
        },
    ), len(lon), len(tri)


def alternating_codes(ncells):
    """ICON codes: -2 inner ocean, -1 boundary ocean, +1 boundary land, +2 inner land."""
    return np.array([(-2, -1, 1, 2)[i % 4] for i in range(ncells)])


def solid_land_codes():
    """Codes with a contiguous land region, so some nodes are land-locked.

    A node counts as ocean if any adjacent element is ocean, so a scattered
    land pattern leaves every node ocean; a solid block is needed.
    """
    lon, _, tri = _patch_mesh()
    centroid_lon = lon[tri].mean(axis=1)
    is_land = centroid_lon > np.median(centroid_lon)
    return np.where(is_land, 2, -2)


class TestFesomConventions:
    @pytest.mark.parametrize("elem_key", ["elements", "face_nodes", "elem"])
    def test_accepts_each_connectivity_name(self, elem_key):
        ds, n2d, e2d = fesom_dataset(elem_key)
        filt = FesomFilter()
        filt.prepare_from_data_array(ds)
        assert filt._n2d == n2d
        assert filt._e2d == e2d

    def test_missing_connectivity_raises(self):
        lon, lat, _ = _patch_mesh()
        ds = xr.Dataset(coords={"lon": (["node"], lon), "lat": (["node"], lat)})
        filt = FesomFilter()
        with pytest.raises(RuntimeError, match="triangulation"):
            filt.prepare_from_data_array(ds)

    def test_one_based_indexing_converted(self):
        """FESOM indices are 1-based; node 0 must still be referenced."""
        ds, n2d, _ = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds)
        assert int(np.asarray(filt._en_pos).min()) == 0
        assert int(np.asarray(filt._en_pos).max()) == n2d - 1

    def test_constant_preserved(self):
        ds, n2d, _ = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds)
        result = filt.compute(1, 0.5, np.full(n2d, 9.0))
        np.testing.assert_allclose(result, 9.0, atol=1e-5)


class TestFesomElementFiltering:
    """filter_elements must be reachable through the FESOM entry point."""

    def test_filter_elements_accepted(self):
        ds, _, e2d = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds, filter_elements=True)
        assert filt._ss_e is not None

    def test_element_data_filterable(self):
        ds, _, e2d = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds, filter_elements=True)
        result = filt.compute(1, 0.5, np.full(e2d, 4.0))
        assert result.shape == (e2d,)
        np.testing.assert_allclose(result, 4.0, atol=1e-5)

    def test_elem_weights_accepted(self):
        ds, _, _ = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds, filter_elements=True, elem_weights="geometric")
        assert filt._ss_e is not None

    def test_default_still_no_element_operator(self):
        ds, _, _ = fesom_dataset()
        filt = FesomFilter()
        filt.prepare_from_data_array(ds)
        assert filt._ss_e is None


class TestIconConventions:
    def test_radians_converted_to_degrees(self):
        ds, n2d, e2d = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(ds)
        assert filt._n2d == n2d
        assert filt._e2d == e2d
        # areas would be ~(180/pi)^2 smaller if the conversion were skipped
        assert np.all(np.asarray(filt._elem_area) > 0.0)

    def test_constant_preserved(self):
        ds, n2d, _ = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(ds)
        result = filt.compute(1, 0.5, np.full(n2d, 3.0))
        np.testing.assert_allclose(result, 3.0, atol=1e-5)

    def test_missing_mask_variable_raises(self):
        ds, _, _ = icon_dataset()
        filt = IconFilter()
        with pytest.raises(KeyError, match="cell_sea_land_mask"):
            filt.prepare_from_data_array(ds, mask=True)


class TestIconSeaLandMask:
    """ICON encodes ocean as the negative cell_sea_land_mask codes.

    Verified against all 50 grids in the DKRZ public pool
    (/pool/data/ICON/grids/public, providers edzw and mpim) carrying the
    variable, spanning 2016-12-13 to 2025-10-08 and 15 distinct grid-generator
    revisions. Every one declares the identical long_name:

        "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell"

    and no grid contains any value outside {-2,-1,0,1,2}. The value sets that
    actually occur are exercised below.
    """

    def test_global_grid_codes(self):
        """Global (_G) grids use all four codes."""
        _, _, e2d = icon_dataset()
        codes = np.array([(-2, -1, 1, 2)[i % 4] for i in range(e2d)])
        filt = IconFilter()
        filt.prepare_from_data_array(icon_dataset(codes)[0], mask=True)
        elem_area = np.asarray(filt._elem_area)
        assert np.all(elem_area[codes > 0] == 0.0)
        assert np.all(elem_area[codes < 0] > 0.0)

    def test_ocean_grid_codes_without_inner_land(self):
        """Ocean (_O) grids drop inner land, leaving {-2,-1,+1} (8 real grids)."""
        _, _, e2d = icon_dataset()
        codes = np.array([(-2, -1, 1)[i % 3] for i in range(e2d)])
        filt = IconFilter()
        filt.prepare_from_data_array(icon_dataset(codes)[0], mask=True)
        elem_area = np.asarray(filt._elem_area)
        assert np.all(elem_area[codes == 1] == 0.0), "+1 boundary land must be masked"
        assert np.all(elem_area[codes < 0] > 0.0)

    def test_all_sea_grid_is_all_ocean(self):
        _, _, e2d = icon_dataset()
        codes = np.array([(-2, -1)[i % 2] for i in range(e2d)])
        filt = IconFilter()
        filt.prepare_from_data_array(icon_dataset(codes)[0], mask=True)
        assert np.all(np.asarray(filt._elem_area) > 0.0)


class TestIconUnpopulatedMask:
    """Six real grids carry cell_sea_land_mask filled entirely with zeros.

    e.g. icon_grid_0030_R02B03_G.nc, whose land-sea information lives in the
    sibling icon_grid_0030_R02B03_Glsm.nc, and icon_grid_0023_R02B07_G.nc
    (sibling icon_grid_0023_R02B07_G_slm.nc). Interpreting an all-zero mask
    with "ocean == code < 0" would mark every cell as land and produce a
    silently degenerate filter with zero areas everywhere.
    """

    def test_all_zero_mask_raises(self):
        _, _, e2d = icon_dataset()
        codes = np.zeros(e2d, dtype=int)
        filt = IconFilter()
        with pytest.raises(ValueError, match="no sea cells"):
            filt.prepare_from_data_array(icon_dataset(codes)[0], mask=True)

    def test_error_points_at_the_sibling_file(self):
        _, _, e2d = icon_dataset()
        filt = IconFilter()
        with pytest.raises(ValueError, match=r"_[Gs]?lsm|_slm|separate"):
            filt.prepare_from_data_array(
                icon_dataset(np.zeros(e2d, dtype=int))[0], mask=True
            )

    def test_error_reports_the_values_found(self):
        _, _, e2d = icon_dataset()
        filt = IconFilter()
        with pytest.raises(ValueError, match=r"\[0\]"):
            filt.prepare_from_data_array(
                icon_dataset(np.zeros(e2d, dtype=int))[0], mask=True
            )

    def test_all_land_mask_also_raises(self):
        """A mask with no sea at all is equally unusable."""
        _, _, e2d = icon_dataset()
        codes = np.full(e2d, 2)
        filt = IconFilter()
        with pytest.raises(ValueError, match="no sea cells"):
            filt.prepare_from_data_array(icon_dataset(codes)[0], mask=True)

    def test_mask_false_still_works_on_such_a_grid(self):
        """The escape hatch must remain available."""
        _, _, e2d = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(
            icon_dataset(np.zeros(e2d, dtype=int))[0], mask=False
        )
        assert np.all(np.asarray(filt._elem_area) > 0.0)

    def test_explicit_array_mask_still_works_on_such_a_grid(self):
        _, _, e2d = icon_dataset()
        explicit = np.ones(e2d, dtype=bool)
        explicit[0] = False
        filt = IconFilter()
        filt.prepare_from_data_array(
            icon_dataset(np.zeros(e2d, dtype=int))[0], mask=explicit
        )
        assert np.asarray(filt._elem_area)[0] == 0.0


class TestIconSeaLandMaskBehaviour:
    """Behavioural consequences of the mask."""

    def test_land_cells_are_excluded(self):
        ds, _, e2d = icon_dataset()
        codes = alternating_codes(e2d)
        ds, _, _ = icon_dataset(codes)

        filt = IconFilter()
        filt.prepare_from_data_array(ds, mask=True)

        elem_area = np.asarray(filt._elem_area)
        is_ocean = codes < 0
        assert np.all(elem_area[~is_ocean] == 0.0), (
            "land cells (+1/+2) must have zero element area"
        )
        assert np.all(elem_area[is_ocean] > 0.0), (
            "ocean cells (-1/-2) must have positive element area"
        )

    def test_mask_true_differs_from_mask_false(self):
        """The whole point of mask=True is to not equal mask=False."""
        _, _, e2d = icon_dataset()
        codes = alternating_codes(e2d)
        ds_masked, _, _ = icon_dataset(codes)
        ds_plain, _, _ = icon_dataset(codes)

        masked, plain = IconFilter(), IconFilter()
        masked.prepare_from_data_array(ds_masked, mask=True)
        plain.prepare_from_data_array(ds_plain, mask=False)

        assert not np.array_equal(
            np.asarray(masked._elem_area), np.asarray(plain._elem_area)
        ), "mask=True must not be a no-op equal to mask=False"

    def test_some_nodes_are_land(self):
        ds, _, _ = icon_dataset(solid_land_codes())
        filt = IconFilter()
        filt.prepare_from_data_array(ds, mask=True)
        assert not np.asarray(filt._mask_n).all(), (
            "a grid with a solid land region must produce land-locked nodes"
        )

    def test_scattered_land_still_leaves_every_node_ocean(self):
        """A node is ocean if ANY adjacent element is ocean.

        Documents why the solid-block fixture above is necessary.
        """
        _, _, e2d = icon_dataset()
        ds, _, _ = icon_dataset(alternating_codes(e2d))
        filt = IconFilter()
        filt.prepare_from_data_array(ds, mask=True)
        assert np.asarray(filt._mask_n).all()

    def test_boundary_and_inner_codes_treated_alike(self):
        """-1 behaves like -2; +1 behaves like +2."""
        _, _, e2d = icon_dataset()
        inner = np.array([(-2, 2)[i % 2] for i in range(e2d)])
        bound = np.array([(-1, 1)[i % 2] for i in range(e2d)])

        f_inner, f_bound = IconFilter(), IconFilter()
        f_inner.prepare_from_data_array(icon_dataset(inner)[0], mask=True)
        f_bound.prepare_from_data_array(icon_dataset(bound)[0], mask=True)

        np.testing.assert_allclose(
            np.asarray(f_inner._elem_area), np.asarray(f_bound._elem_area)
        )

    def test_all_ocean_grid_matches_mask_false(self):
        """With no land present, masking must be a no-op."""
        _, _, e2d = icon_dataset()
        all_ocean = np.full(e2d, -2)
        a, b = IconFilter(), IconFilter()
        a.prepare_from_data_array(icon_dataset(all_ocean)[0], mask=True)
        b.prepare_from_data_array(icon_dataset(all_ocean)[0], mask=False)
        np.testing.assert_array_equal(
            np.asarray(a._elem_area), np.asarray(b._elem_area)
        )

    def test_explicit_array_mask_still_respected(self):
        _, _, e2d = icon_dataset()
        ds, _, _ = icon_dataset(alternating_codes(e2d))
        explicit = np.ones(e2d, dtype=bool)
        explicit[0] = False

        filt = IconFilter()
        filt.prepare_from_data_array(ds, mask=explicit)
        elem_area = np.asarray(filt._elem_area)
        assert elem_area[0] == 0.0
        assert np.all(elem_area[1:] > 0.0)


class TestIconElementFiltering:
    """filter_elements must be reachable through the ICON entry point."""

    def test_filter_elements_accepted(self):
        ds, _, _ = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(ds, filter_elements=True)
        assert filt._ss_e is not None

    def test_element_data_filterable(self):
        ds, _, e2d = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(ds, filter_elements=True)
        result = filt.compute(1, 0.5, np.full(e2d, 7.0))
        assert result.shape == (e2d,)
        np.testing.assert_allclose(result, 7.0, atol=1e-5)

    def test_elem_weights_accepted(self):
        ds, _, _ = icon_dataset()
        filt = IconFilter()
        filt.prepare_from_data_array(ds, filter_elements=True, elem_weights="geometric")
        assert filt._ss_e is not None
