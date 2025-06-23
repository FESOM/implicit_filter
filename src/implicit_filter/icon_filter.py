import xarray as xr
import numpy as np

from .triangular_filter import TriangularFilter


class IconFilter(TriangularFilter):
    def prepare_from_file(
        self,
        grid_file: str,
        ocean_mask: np.ndarray | bool = False,
        full: bool = False,
        gpu: bool = False,
    ):
        grid2d = xr.open_dataset(grid_file)
        # Prepare the mesh data
        xcoord = grid2d["vlon"].values * 180.0 / np.pi
        ycoord = grid2d["vlat"].values * 180.0 / np.pi  # Location of nodes, in degrees
        tri = grid2d["vertex_of_cell"].values.T.astype(int) - 1
        tri = tri.astype(int)

        if isinstance(ocean_mask, np.ndarray):
            pass
        elif ocean_mask:
            if "cell_sea_land_mask" in grid2d:
                ocean_mask = grid2d["cell_sea_land_mask"].values
            else:
                raise KeyError(
                    f"In the file {grid_file} there's no ocean mask under default name {'cell_sea_land_mask'}"
                )
        else:
            ocean_mask = np.ones(len(tri[:, 1]))
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
            mask=ocean_mask,
            gpu=gpu,
        )
