import numpy as np
import xarray as xr

from implicit_filter.utils._auxiliary import find_adjacent_points_north
from implicit_filter.utils._numpy_functions import calculate_global_nemo_neighbourhood
from .latlon_filter import LatLonFilter


class NemoFilter(LatLonFilter):
    """
    A filter class for NEMO ocean model data using NumPy arrays.
    """

    def prepare_from_file(
        self,
        file: str,
        vl: int,
        mask: np.ndarray | bool = True,
        gpu: bool = False,
    ):
        ds = xr.open_dataset(file)

        nx, ny = (
            ds.gphit.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values.shape
        )
        north_adj, _ = find_adjacent_points_north(file, 1e-5)
        e2d = nx * ny

        self._nx = nx
        self._ny = ny
        self._e2d = e2d

        ee_pos, nza = calculate_global_nemo_neighbourhood(e2d, nx, ny, north_adj)
        self._ee_pos = ee_pos

        # Cell sizes
        hx = np.reshape(
            ds.e1t.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        hy = np.reshape(
            ds.e2t.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        self._area = hx * hy

        hh = np.ones((4, e2d))  # Edge lengths
        hh[1, :] = np.reshape(
            ds.e2u.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # North edge
        hh[0, :] = np.reshape(
            ds.e1v.isel(t=0, y=slice(None, -2), x=slice(None, -2))
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
            ds.e3u_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3v = np.reshape(
            ds.e3v_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )
        h3t = np.reshape(
            ds.e3t_0.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )

        hc = np.ones((4, e2d))  # Distance to next cell centers
        hc[0, :] = np.reshape(
            ds.e1u.isel(t=0, y=slice(None, -2), x=slice(None, -2))
            .transpose("x", "y")
            .values
            / 1000.0,
            nx * ny,
        )  # West neighbour
        hc[1, :] = np.reshape(
            ds.e2v.isel(t=0, y=slice(None, -2), x=slice(None, -2))
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
                ds.tmask.isel(t=0, z=vl, y=slice(None, -2), x=slice(None, -2))
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
                        (hh[m, n] * h3u[n]) / (hc[m, n] * h3t[n])
                        if m % 2 == 0
                        else (hh[m, n] * h3v[n]) / (hc[m, n] * h3t[n])
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

        self.set_backend("gpu" if gpu else "cpu")
