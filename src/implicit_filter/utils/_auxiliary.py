import math
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pandas import Series
from sklearn.linear_model import LinearRegression


def neighboring_triangles(n2d: int, e2d: int, tri: np.ndarray):
    """
    Calculate neighboring triangles for each node in a 2D mesh.

    Parameters:
    ----------
    n2d : int
        The total number of nodes in the mesh.

    e2d : int
        The total number of triangles (elements) in the mesh.

    tri : np.ndarray
        A 2D NumPy array representing the connectivity of triangles (elements) in the mesh.
        Each row contains the indices of the three nodes that form a triangle.

    Returns:
    -------
    ne_num : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the count of neighboring triangles for each node.

    ne_pos : np.ndarray
        A 2D NumPy array of shape (max_neighboring_triangles, n2d) containing the positions of neighboring
        triangles for each node. The 'ne_pos[i, j]' entry indicates the index of the i-th neighboring triangle
        for the j-th node. Unused entries are filled with zeros.
    """
    # Initialize an array to store the count of neighboring triangles for each node.
    ne_num = np.zeros([n2d], dtype=int)

    # Loop through each triangle (element) in the mesh.
    for n in range(e2d):
        enodes = tri[n, :]
        # Increment the count of neighboring triangles for each node in the current triangle.
        ne_num[enodes] += 1

    # Initialize an array to store the positions of neighboring triangles for each node.
    ne_pos = np.zeros([int(np.max(ne_num)), n2d], dtype=int)

    # Reset the array to store the count of neighboring triangles for each node.
    ne_num = np.zeros([n2d], dtype=int)

    # Loop through each triangle (element) in the mesh.
    for n in range(e2d):
        enodes = tri[n, :]
        # Loop through the nodes of the current triangle.
        for j in range(3):
            # Store the position of the current neighboring triangle for the corresponding node.
            ne_pos[ne_num[enodes[j]], enodes[j]] = n
            # Increment the count of neighboring triangles for the node.
        ne_num[enodes] += 1

    return ne_num, ne_pos


def neighbouring_nodes(
    n2d: int, tri: np.ndarray, ne_num: np.ndarray, ne_pos: np.ndarray
):
    """
    Compute neighboring nodes for each node in a 2D mesh.

    Parameters:
    ----------
    n2d : int
        The total number of nodes in the mesh.

    tri : np.ndarray
        A 2D NumPy array representing the connectivity of triangles (elements) in the mesh.
        Each row contains the indices of the three nodes that form a triangle.

    ne_num : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the count of neighboring triangles for each node.

    ne_pos : np.ndarray
        A 2D NumPy array of shape (max_neighboring_triangles, n2d) containing the positions of neighboring
        triangles for each node. The 'ne_pos[i, j]' entry indicates the index of the i-th neighboring triangle
        for the j-th node. Unused entries are filled with zeros.

    Returns:
    -------
    nn_num : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the number of neighboring nodes for each node.

    nn_pos : np.ndarray
        A 2D NumPy array of shape (max_neighboring_nodes, n2d) containing the positions of neighboring nodes
        for each node. The 'nn_pos[i, j]' entry indicates the index of the i-th neighboring node for the j-th node.
    """
    # Initialize an array to store the count of neighboring nodes for each node.

    # Initialize an array to store the positions of neighboring nodes for each node.
    nn_num = np.zeros([n2d], dtype=int)
    check = np.zeros([n2d], dtype=int)
    aux = np.zeros([10], dtype=int)
    for j in range(n2d):
        cc = 0
        for m in range(ne_num[j]):
            el = ne_pos[m, j]
            for k in range(3):
                a = tri[el, k]
                if check[a] == 0:
                    check[a] = 1
                    aux[cc] = a
                    cc += 1

        nn_num[j] = cc
        check[aux[0:cc]] = 0

    nn_pos = np.zeros([np.max(nn_num), n2d], dtype=int)

    for j in range(n2d):
        cc = 0
        for m in range(ne_num[j]):
            el = ne_pos[m, j]
            for k in range(3):
                a = tri[el, k]
                if check[a] == 0:
                    check[a] = 1
                    aux[cc] = a
                    cc += 1

        nn_pos[0:cc, j] = aux[0:cc].T
        check[aux[0:cc]] = 0

    return nn_num, nn_pos


def areas(
    n2d: int,
    e2d: int,
    tri: np.ndarray,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    ne_num: np.ndarray,
    ne_pos: np.ndarray,
    meshtype: str,
    carthesian: bool,
    cyclic_length: float,
    mask: np.ndarray,
):
    """
    Calculate areas of triangles and derivatives of P1 basis functions.

    Parameters:
    ----------
    n2d : int
        The total number of nodes in the mesh.

    e2d : int
        The total number of triangles (elements) in the mesh.

    tri : np.ndarray
        A 2D NumPy array representing the connectivity of triangles (elements) in the mesh.
        Each row contains the indices of the three nodes that form a triangle.

    xcoord : np.ndarray
        A 1D NumPy array containing the x-coordinates of nodes, in the mesh.

    ycoord : np.ndarray
        A 1D NumPy array containing the y-coordinates of nodes in the mesh.

    ne_num : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the count of neighboring triangles for each node.

    ne_pos : np.ndarray
        A 2D NumPy array of shape (max_neighboring_triangles, n2d) containing the positions of neighboring
        triangles for each node. The 'ne_pos[i, j]' entry indicates the index of the i-th neighboring triangle
        for the j-th node. Unused entries are filled with zeros.

    meshtype : str
        Mesh type, either 'm' (metric) or 'r' (radial).

    carthesian : bool
        Boolean indicating whether the mesh is in Cartesian coordinates.

    cyclic_length : float
        The length of the cyclic boundary if the mesh is cyclic (for 'r' meshtype).

    Returns:
    -------
    area : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the scalar cell (cluster) area for each node.

    elem_area : np.ndarray
        A 1D NumPy array of shape (e2d,) containing the area of each triangle (element) in the mesh.

    dx : np.ndarray
        A 2D NumPy array of shape (e2d, 3) containing the x-derivative of P1 basis functions for each triangle.

    dy : np.ndarray
        A 2D NumPy array of shape (e2d, 3) containing the y-derivative of P1 basis functions for each triangle.

    Mt : np.ndarray
        A 1D NumPy array of shape (e2d,) containing a factor for metric terms based on meshtype and coordinates.
    """
    dx = np.zeros([e2d, 3], dtype=float)
    dy = np.zeros([e2d, 3], dtype=float)
    elem_area = np.zeros([e2d])
    r_earth = 6400  # Earth's radius, assuming units in kilometers
    Mt = np.ones([e2d])

    if meshtype == "m":
        for n in range(e2d):
            # Calculate differences in x and y coordinates for triangle vertices.
            x2 = xcoord[tri[n, 1]] - xcoord[tri[n, 0]]
            x3 = xcoord[tri[n, 2]] - xcoord[tri[n, 0]]
            y2 = ycoord[tri[n, 1]] - ycoord[tri[n, 0]]
            y3 = ycoord[tri[n, 2]] - ycoord[tri[n, 0]]

            # Calculate determinant of the Jacobian matrix for this triangle.
            d = x2 * y3 - y2 * x3

            # Calculate x and y derivatives of P1 basis functions.
            dx[n, 0] = (-y3 + y2) / d
            dx[n, 1] = y3 / d
            dx[n, 2] = -y2 / d

            dy[n, 0] = -(-x3 + x2) / d
            dy[n, 1] = -x3 / d
            dy[n, 2] = x2 / d

            # Calculate the area of the triangle.
            elem_area[n] = 0.5 * abs(d) * mask[n]

    elif meshtype == "r":
        rad = math.pi / 180.0
        if carthesian:
            Mt = np.ones([e2d])
        else:
            Mt = np.cos(np.sum(rad * ycoord[tri], axis=1) / 3.0)

        for n in range(e2d):
            # Calculate differences in longitude and latitude for triangle vertices.
            x2 = rad * (xcoord[tri[n, 1]] - xcoord[tri[n, 0]])
            x3 = rad * (xcoord[tri[n, 2]] - xcoord[tri[n, 0]])
            y2 = r_earth * rad * (ycoord[tri[n, 1]] - ycoord[tri[n, 0]])
            y3 = r_earth * rad * (ycoord[tri[n, 2]] - ycoord[tri[n, 0]])

            # Adjust for cyclic boundaries.
            if x2 > cyclic_length / 2.0:
                x2 = x2 - cyclic_length
            if x2 < -cyclic_length / 2.0:
                x2 = x2 + cyclic_length
            if x3 > cyclic_length / 2.0:
                x3 = x3 - cyclic_length
            if x3 < -cyclic_length / 2.0:
                x3 = x3 + cyclic_length

            # Apply metric factors and calculate x and y derivatives of P1 basis functions.
            x2 = r_earth * x2 * Mt[n]
            x3 = r_earth * x3 * Mt[n]
            d = x2 * y3 - y2 * x3

            dx[n, 0] = (-y3 + y2) / d
            dx[n, 1] = y3 / d
            dx[n, 2] = -y2 / d

            dy[n, 0] = -(-x3 + x2) / d
            dy[n, 1] = -x3 / d
            dy[n, 2] = x2 / d

            # Calculate the area of the triangle.
            elem_area[n] = 0.5 * abs(d) * mask[n]

        if carthesian:
            Mt = np.zeros([e2d])
        else:
            Mt = (np.sin(rad * np.sum(ycoord[tri], axis=1) / 3.0) / Mt) / r_earth

    # Calculate scalar cell (cluster) area for each node.
    area = np.zeros([n2d])
    for n in range(n2d):
        area[n] = np.sum(elem_area[ne_pos[0 : ne_num[n], n]]) / 3.0
        if area[n] == 0.0:
            area[n] = 1.0e-5

    return area, elem_area, dx, dy, Mt


def make_tri(nodnum: np.ndarray, nx: int, ny: int):
    """
    Compute the triangulation for mock data using the given node numbering, width, and height.

    This function generates triangles that form a mesh over a rectangular domain defined by the mock data's
    node numbering, width (nx), and height (ny).

    Parameters:
    ----------
    nodnum : np.ndarray
        A 2D NumPy array of shape (ny, nx) containing the node numbering.

    nx : int
        Width of the domain (number of nodes in the x-direction).

    ny : int
        Height of the domain (number of nodes in the y-direction).

    Returns:
    -------
    np.ndarray
        A 2D NumPy array of shape (2 * (nx - 1) * (ny - 1), 3) representing the triangulation of the mock data.
        Each row contains the indices of three nodes that form a triangle.

    Notes:
    ------
    This function assumes that the nodnum array provides a consistent node numbering for a rectangular domain.
    The generated triangulation covers the domain with triangles formed by adjacent nodes.

    Example:
    --------
    nodnum = np.array([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]])
    nx = 3
    ny = 3
    tri = make_tri(nodnum, nx, ny)
    # Resulting tri array:
    # array([[0, 3, 1],
    #        [3, 4, 1],
    #        [1, 4, 2],
    #        [4, 5, 2],
    #        [3, 6, 4],
    #        [6, 7, 4],
    #        [4, 7, 5],
    #        [7, 8, 5]])
    """
    tri = np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=int)
    mx = 0
    for n in range(nx - 1):
        for nn in range(ny - 1):
            tri[mx, :] = [nodnum[nn, n], nodnum[nn + 1, n], nodnum[nn, n + 1]]
            mx += 1
            tri[mx, :] = [nodnum[nn + 1, n], nodnum[nn + 1, n + 1], nodnum[nn, n + 1]]
            mx += 1

    return tri


def convert_to_wavenumbers(dist, dxm):
    """
    Converts a given spatial distance to wavenumbers for spectral analysis.

    Parameters:
    -----------
    dist : float
        The spatial distance for which wavenumbers are to be calculated.
    dxm : float
        The mesh resolution, representing the distance between grid points.

    Returns:
    --------
    float
        The corresponding wavenumber for the given spatial distance.

    Notes:
    ------
    - Input parameters `dist` and `dxm` must have the same unit.
    - The factor 3.5 is used to make the results comparable with box-type filters.

    """
    if np.any(np.logical_or(dist <= 0, dxm <= 0)):
        raise ValueError("Both dist and dxm parameters must be positive")

    size = 3.5 * (dist / dxm)
    return 2 * math.pi / size


def find_adjacent_points_north(
    ds_mm: xr.DataArray = None, lon_lat_prec_degrees: float = None
) -> tuple[Series, int]:
    """
    Fix rounding erros in NEMO grid using linear regression

    Author: Willi Rath

    Parameters:
    -----------
    ds_mm : xr.DataArray
        Path to the mesh mask file.
    lon_lat_prec_degrees : float
        Rounding precision for longitude and latitude coordinates.

    Returns:
    --------
    pd.Series
        Pandas Series containing the indices of adjacent points at northern border.

    """
    # load mesh mask
    ds_mm = ds_mm.squeeze(drop=True)
    ds_mm = ds_mm.assign_coords(
        x=np.arange(ds_mm.sizes["x"]),
        y=np.arange(ds_mm.sizes["y"]),
    )

    # extract T-point lon and lat
    lon, lat = ds_mm.glamt, ds_mm.gphit

    # Cast lon lat to int
    ilon = (lon / lon_lat_prec_degrees).astype(int)
    ilat = (lat / lon_lat_prec_degrees).astype(int)

    # Find out which row corresponds to the last one y
    if (
        abs(np.sort(ilon.isel(y=-1)) - np.sort(ilon.isel(y=-2))).mean()
        < abs(np.sort(ilon.isel(y=-1)) - np.sort(ilon.isel(y=-3))).mean()
    ):
        corresponds_to_redundant = -2
    else:
        corresponds_to_redundant = -3
    # print("corresponding row is at y = ", corresponds_to_redundant)

    # extract redundant (last row) and correspoding row coords
    ilon_redundant = ilon.isel(x=slice(1, -1)).isel(y=-1, drop=True)
    ilon_corresponds = ilon.isel(x=slice(1, -1)).isel(
        y=corresponds_to_redundant, drop=True
    )
    ilat_redundant = ilon.isel(x=slice(1, -1)).isel(y=-1, drop=True)
    ilat_corresponds = ilon.isel(x=slice(1, -1)).isel(
        y=corresponds_to_redundant, drop=True
    )

    # find corresponding x
    x_corr = []
    for x_r, lon_r, lat_r in list(
        zip(ilon_redundant.x.data, ilon_redundant.data, ilat_redundant.data)
    ):
        for x_c, lon_c, lat_c in zip(
            ilon_corresponds.x.data, ilon_corresponds.data, ilat_corresponds.data
        ):
            if lon_r.data[()] == lon_c.data[()]:
                if lat_r.data[()] == lat_c.data[()]:
                    if x_c not in x_corr:
                        x_corr.append(x_c)
                        break

    # filter adjacent x for outliers
    adjacent_x = pd.Series(
        x_corr,
        name="adjacent_x",
        index=pd.Series(ilon_redundant.x.data, name="reference_x"),
    )
    adjacent_x_sanitized = adjacent_x.where(
        abs(adjacent_x.diff()) < 1.5 * abs(adjacent_x.diff()).mean()
    ).dropna()

    # fit clean adjacent indices
    lr = LinearRegression()
    lr.fit(
        np.array(adjacent_x_sanitized.index).reshape(-1, 1),
        np.array(adjacent_x_sanitized).reshape(-1, 1),
    )
    adjacent_x_fit = (
        pd.Series(
            lr.predict(np.array(adjacent_x.index).reshape(-1, 1)).reshape(-1),
            index=adjacent_x.index,
            name="adjacent_x",
        )
        .round()
        .astype(int)
    )

    # test if we were successful
    np.testing.assert_array_almost_equal(
        ds_mm.glamt.isel(y=corresponds_to_redundant).sel(x=adjacent_x_fit.to_xarray()),
        ds_mm.glamt.isel(y=-1).sel(x=adjacent_x_fit.index.to_series().to_xarray()),
    )

    # return adjacent indices
    return adjacent_x_fit, corresponds_to_redundant
