import xarray as xr
import numpy as np

from .triangular_filter import TriangularFilter


class IconFilter(TriangularFilter):
    def prepare_from_file(
        self,
        grid_file: str,
        full: bool = False,
        mask: np.ndarray | bool = False,
        gpu: bool = False,
    ):
        grid2d = xr.open_dataset(grid_file)
        # Prepare the mesh data
        xcoord = grid2d["vlon"].values * 180.0 / np.pi
        ycoord = grid2d["vlat"].values * 180.0 / np.pi  # Location of nodes, in degrees
        tri = grid2d["vertex_of_cell"].values.T.astype(int) - 1
        tri = tri.astype(int)
        mask_transform = False

        if isinstance(mask, np.ndarray):
            pass
        elif mask:
            if "cell_sea_land_mask" in grid2d:
                mask = grid2d["cell_sea_land_mask"].values
                mask_transform = True
            else:
                raise KeyError(
                    f"In the file {grid_file} there's no ocean mask under default name {'cell_sea_land_mask'}"
                )
        else:
            mask = np.ones(len(tri[:, 1]))
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
            mask_transform=mask_transform,
        )
