import xarray as xr
import numpy as np

from .triangular_filter import TriangularFilter


class IconFilter(TriangularFilter):
    def prepare_ICON_filter(
        self,
        grid2d: xr.DataArray,
        ocean_mask: xr.DataArray = None,
        full: bool = False,
        gpu: bool = False,
    ):

        # Prepare the mesh data
        xcoord = grid2d["vlon"].values * 180.0 / np.pi
        ycoord = grid2d["vlat"].values * 180.0 / np.pi  # Location of nodes, in degrees
        tri = grid2d["vertex_of_cell"].values.T - 1
        tri = tri.astype(int)

        if ocean_mask is None:
            mask = np.ones(len(tri[:, 1]))
        else:
            # mask = xr.where(ocean_mask.values < 0.0, 1.0, 0.0)
            mask = ocean_mask.values
            # NOTE: LSM is in grid2d['cell_sea_land_mask'] or in grid3d['lsm_c'].isel(depth=???)

        self.prepare(
            len(xcoord),
            len(tri[:, 1]),
            tri,
            xcoord,
            ycoord,
            meshtype="r",
            cartesian=False,
            cyclic_length=2.0 * np.pi,
            full=full,
            mask=mask,
            gpu=gpu,
        )
