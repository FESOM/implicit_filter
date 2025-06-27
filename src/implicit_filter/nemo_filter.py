import numpy as np
import xarray as xr
import math

from implicit_filter.utils._auxiliary import find_adjacent_points_north
from implicit_filter.utils._numpy_functions import (
    calculate_global_nemo_neighbourhood,
    calculate_global_regular_neighbourhood,
    calculate_local_regular_neighbourhood,
)
from .latlon_filter import LatLonFilter


class NemoFilter(LatLonFilter):
    """
    Filter implementation specialized for NEMO ocean model grids.

    This class extends LatLonFilter to handle NEMO's specific grid characteristics
    including partial cells, scale factors, and complex boundary representations.
    It supports different neighborhood configurations for accurate filtering.

    Parameters
    ----------
    See LatLonFilter for inherited parameters.

    Notes
    -----
    - Automatically handles NEMO's redundant point representation
    - Supports three neighborhood types: 'full', 'west-east', and 'local'
    - Accounts for partial cells through 3D scale factors
    """

    def prepare_from_file(
        self,
        file: str,
        vl: int,
        mask: np.ndarray | bool = True,
        gpu: bool = False,
        neighb: str = "full",
    ):
        """
        Configure filter using a NEMO grid file.

        Parameters
        ----------
        file : str
            Path to NEMO grid file (NetCDF format).
        vl : int
            Vertical level index for which to configure the filter.
        mask : np.ndarray | bool, optional
            Land-sea mask specification:
            - np.ndarray: Precomputed mask array
            - True: Auto-detect from 'tmask' variable (default)
            - False: All ocean cells
        gpu : bool, optional
            True to enable GPU acceleration (default: False).
        neighb : str, optional
            Neighborhood type:
            - 'full': Full 4-point neighborhood with North Pole handling
            - 'west-east': Zonal connections only
            - 'local': Standard 4-point neighborhood (default: 'full')

        Notes
        -----
        - Automatically detects and handles NEMO's redundant grid points
        - Converts grid metrics from meters to kilometers
        - Uses vertical level scale factors for accurate cell volumes
        """
        ds = xr.open_dataset(file)
        self.prepare_from_data_array(ds, vl, mask, gpu, neighb)

    def prepare_from_data_array(
        self,
        ds: xr.DataArray,
        vl: int,
        mask: np.ndarray | bool = True,
        gpu: bool = False,
        neighb: str = "full",
    ):
        """
        Configure filter using an xarray Dataset containing NEMO grid data.

        Parameters
        ----------
        ds : xr.Dataset
            xarray Dataset containing NEMO grid variables.
        vl : int
            Vertical level index for filter configuration.
        mask : np.ndarray | bool, optional
            Land-sea mask specification (see prepare_from_file).
        gpu : bool, optional
            GPU acceleration flag (default: False).
        neighb : str, optional
            Neighborhood type (see prepare_from_file).

        Raises
        ------
        NotImplementedError
            If unsupported neighborhood type is specified.

        Notes
        -----
        - Requires standard NEMO grid variables: gphit, e1t, e2t, e1u, e2v, e3u_0, e3v_0, e3t_0
        - The 'tmask' variable is used for land-sea masking when auto-detection is enabled
        - Handles North Pole folding in 'full' neighborhood configuration
        - Grid metrics are converted from meters to kilometers for consistency
        """
        north_adj = None

        if neighb == "full":
            north_adj, corresponds_to_redundant = find_adjacent_points_north(ds, 1e-5)
        else:
            corresponds_to_redundant = None

        nx, ny = (
            ds.gphit.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values.shape
        )
        e2d = nx * ny

        self._nx = nx
        self._ny = ny
        self._e2d = e2d

        if neighb == "full":
            ee_pos, nza = calculate_global_nemo_neighbourhood(e2d, nx, ny, north_adj)
        elif neighb == "west-east":
            ee_pos, nza = calculate_global_regular_neighbourhood(e2d, nx, ny)
        elif neighb == "local":
            ee_pos, nza = calculate_local_regular_neighbourhood(e2d, nx, ny)
        else:
            raise NotImplementedError(
                f"the neighbourhood type {neighb} is not supported. The only options are full, west-east, local."
            )

        self._ee_pos = ee_pos

        # Cell sizes
        hx = np.reshape(
            ds.e1t.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        hy = np.reshape(
            ds.e2t.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        self._area = hx * hy

        hh = np.ones((4, e2d))  # Edge lengths
        hh[1, :] = np.reshape(
            ds.e2u.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # North edge
        hh[0, :] = np.reshape(
            ds.e1v.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # West edge
        for n in range(e2d):
            if ee_pos[3, n] != n:
                hh[3, n] = hh[1, ee_pos[3, n]]
            else:
                hh[3, n] = hh[1, n]

            if ee_pos[2, n] != n:
                hh[2, n] = hh[0, ee_pos[2, n]]
            else:
                hh[2, n] = hh[0, n]

        # Cell heights
        h3u = np.reshape(
            ds.e3u_0.isel(
                z=vl,
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3v = np.reshape(
            ds.e3v_0.isel(
                z=vl,
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3t = np.reshape(
            ds.e3t_0.isel(
                z=vl,
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )

        hc = np.ones((4, e2d))  # Distance to next cell centers
        hc[0, :] = np.reshape(
            ds.e1u.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # West neighbour
        hc[1, :] = np.reshape(
            ds.e2v.isel(
                y=slice(None, corresponds_to_redundant),
                x=slice(None, corresponds_to_redundant),
            )
            .squeeze()
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # North neighbour

        for n in range(e2d):
            if ee_pos[3, n] != n:
                hc[3, n] = hc[1, ee_pos[1, n]]
            else:
                hc[3, n] = hc[1, n]

            if ee_pos[2, n] != n:
                hc[2, n] = hh[2, ee_pos[2, n]]
            else:
                hc[2, n] = hc[2, n]

        ss = np.zeros(nza, dtype="float")
        ii = np.zeros(nza, dtype="int")
        jj = np.zeros(nza, dtype="int")

        if isinstance(mask, np.ndarray):
            pass
        elif mask:
            mask = np.reshape(
                ds.tmask.isel(
                    z=vl,
                    y=slice(None, corresponds_to_redundant),
                    x=slice(None, corresponds_to_redundant),
                )
                .squeeze()
                .transpose("x", "y")
                .values,
                nx * ny,
            )
        else:
            mask = np.ones(nx * ny, dtype=bool)

        nn = 0
        for n in range(e2d):
            no = nn
            for m in range(4):
                if ee_pos[m, n] != n and mask[ee_pos[m, n]] != 0:
                    nn += 1
                    ss[nn] = (
                        (hh[m, n] * h3u[ee_pos[m, n]]) / (hc[m, n] * h3t[ee_pos[m, n]])
                        if m % 2 == 0
                        else (hh[m, n] * h3v[ee_pos[m, n]])
                        / (hc[m, n] * h3t[ee_pos[m, n]])
                    )

                    ss[nn] /= self._area[n]  # Add division on cell area if you prefer
                    ii[nn] = n
                    jj[nn] = ee_pos[m, n]

            ii[no] = n
            jj[no] = n
            ss[no] = -np.sum(ss[no : nn + 1])
            nn += 1

        self._ss = ss
        self._ii = ii
        self._jj = jj

        mask_sp = np.logical_and(mask[ii], mask[jj])

        self._ss = self._ss[mask_sp]
        self._ii = self._ii[mask_sp]
        self._jj = self._jj[mask_sp]

        self.set_backend("gpu" if gpu else "cpu")
