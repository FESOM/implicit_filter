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
        """
        Configure filter using an ICON grid file.

        Parameters
        ----------
        grid_file : str
            Path to ICON grid file (NetCDF format).
        full : bool, optional
            True to include metric terms in operator (default: False).
        mask : np.ndarray | bool, optional
            Can be:
            - Precomputed land mask array
            - True to automatically detect mask from file
            - False to treat all cells as ocean (default)
        gpu : bool, optional
            True to enable GPU acceleration (default: False).

        """
        grid2d = xr.open_dataset(grid_file)
        self.prepare_from_data_array(grid2d, full, mask, gpu)

    def prepare_from_data_array(
        self,
        grid2d: xr.DataArray,
        full: bool = False,
        mask: np.ndarray | bool = False,
        gpu: bool = False,
    ):
        """
        Configure filter using an xarray Dataset containing ICON grid data.

        Parameters
        ----------
        grid2d : xr.Dataset
            xarray Dataset containing ICON grid variables.
        full : bool, optional
            Metric terms inclusion flag (default: False).
        mask : np.ndarray | bool, optional
            Land-sea mask specification:
            - np.ndarray: Precomputed mask array where true is ocean
            - True: Auto-detect from 'cell_sea_land_mask'
            - False: All ocean cells (default)
        gpu : bool, optional
            GPU acceleration flag (default: False).

        Raises
        ------
        KeyError
            If mask=True but 'cell_sea_land_mask' not found in dataset.

        Notes
        -----
        - Vertex coordinates are converted from radians to degrees
        - Element connectivity is expected in 'vertex_of_cell' variable
        - Spherical coordinates with cyclic domain are assumed
        - Land-sea mask is transformed from cell-centered to nodal representation
        """
        # Prepare the mesh data
        xcoord = grid2d["vlon"].values * 180.0 / np.pi
        ycoord = grid2d["vlat"].values * 180.0 / np.pi  # Location of nodes, in degrees
        tri = grid2d["vertex_of_cell"].values.T.astype(int) - 1
        tri = tri.astype(int)

        if isinstance(mask, np.ndarray):
            pass
        elif mask:
            if "cell_sea_land_mask" in grid2d:
                mask = grid2d["cell_sea_land_mask"].values * -1
                mask[mask == 2] = 1
                mask[mask == -2] = -1
                mask = grid2d["cell_sea_land_mask"].values * -1
                mask = mask.astype(np.bool)
            else:
                raise KeyError(
                    f"In the file grid file there's no ocean mask under default name 'cell_sea_land_mask'"
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
        )
