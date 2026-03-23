import math
from typing import Tuple

import numpy as np


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
    aux = np.zeros([20], dtype=int)
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
    ds_mm=None, lon_lat_prec_degrees: float = None
) -> tuple:
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
    import pandas as pd
    adjacent_x = pd.Series(
        x_corr,
        name="adjacent_x",
        index=pd.Series(ilon_redundant.x.data, name="reference_x"),
    )
    adjacent_x_sanitized = adjacent_x.where(
        abs(adjacent_x.diff()) < 1.5 * abs(adjacent_x.diff()).mean()
    ).dropna()

    # fit clean adjacent indices
    from sklearn.linear_model import LinearRegression
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

def find_and_sort_edges_and_triangles(n2d, nn_num, nn_pos, ne_num, ne_pos):
    """
    Finds unique edges, their associated triangles, and sorts them.
    Internal edges (with two triangles) are placed before boundary edges.
    """
    edge_map = {}
    
    for n in range(n2d):
        for q in range(nn_num[n]):
            node = nn_pos[q, n]
            if node > n:
                n_tris = set(ne_pos[:ne_num[n], n])
                node_tris = set(ne_pos[:ne_num[node], node])
                common_tris = list(n_tris.intersection(node_tris))
                edge_map[(n, node)] = common_tris

    internal_edges, boundary_edges = [], []
    internal_tri, boundary_tri = [], []

    for edge, tris in edge_map.items():
        if len(tris) == 2:
            internal_edges.append(edge)
            internal_tri.append(tris)
        else:
            boundary_edges.append(edge)
            boundary_tri.append([tris[0] if tris else -1, -1])

    sorted_edge_list = internal_edges + boundary_edges
    sorted_tri_list = internal_tri + boundary_tri
    
    edges = np.array(sorted_edge_list).T
    edge_tri = np.array(sorted_tri_list).T
    ed2d_in = len(internal_edges)
    
    return edges, edge_tri, ed2d_in

def calculate_triangle_centers(e2d, tri, xcoord, ycoord, meshtype, cyclic_length):
    """
    Calculates the geometric center (centroid) of each triangle.
    """
    tcenter = np.zeros((2, e2d))
    
    if meshtype == 'm':
        tri_T = tri.T
        tcenter[0, :] = np.mean(xcoord[tri_T], axis=0)
        tcenter[1, :] = np.mean(ycoord[tri_T], axis=0)
        
    elif meshtype == 'r':
        rad = np.pi / 180
        for n in range(e2d):
            elnodes = tri[n, :]
            x_nodes = xcoord[elnodes]
            
            x_diffs = rad * (x_nodes[1:] - x_nodes[0])
            x_diffs[x_diffs > cyclic_length / 2.0] -= cyclic_length
            x_diffs[x_diffs < -cyclic_length / 2.0] += cyclic_length
            x_adjusted = np.concatenate(([x_nodes[0]], x_nodes[0] + x_diffs / rad))
            
            tcenter[0, n] = np.mean(x_adjusted)
            tcenter[1, n] = np.mean(ycoord[elnodes])
        
    return tcenter

def orient_edges(ed2d, edges, edge_tri, tcenter, xcoord, ycoord, meshtype, cyclic_length):
    """
    Orders the direction of edges so that the first triangle is on the left
    of the edge vector.
    """
    edges_oriented = edges.copy()
    edge_tri_oriented = edge_tri.copy()

    for n in range(ed2d):
        ed = edges_oriented[:, n]
        tri1_idx = edge_tri_oriented[0, n]
        
        xc = np.zeros(2)
        xe = np.zeros(2)

        if meshtype == 'm':
            xc[0] = tcenter[0, tri1_idx] - xcoord[ed[0]]
            xc[1] = tcenter[1, tri1_idx] - ycoord[ed[0]]
            xe[0] = xcoord[ed[1]] - xcoord[ed[0]]
            xe[1] = ycoord[ed[1]] - ycoord[ed[0]]
        
        elif meshtype == 'r':
            rad = np.pi / 180
            def get_cyclic_diff(a, b):
                diff = rad * (a - b)
                if diff > cyclic_length / 2.0: 
                    diff -= cyclic_length
                if diff < -cyclic_length / 2.0: 
                    diff += cyclic_length
                return diff / rad

            xc[0] = get_cyclic_diff(tcenter[0, tri1_idx], xcoord[ed[0]])
            xc[1] = tcenter[1, tri1_idx] - ycoord[ed[0]]
            xe[0] = get_cyclic_diff(xcoord[ed[1]], xcoord[ed[0]])
            xe[1] = ycoord[ed[1]] - ycoord[ed[0]]

        if xc[0] * xe[1] - xc[1] * xe[0] > 0:
            if edge_tri_oriented[1, n] != -1:
                edge_tri_oriented[0, n], edge_tri_oriented[1, n] = edge_tri_oriented[1, n], edge_tri_oriented[0, n]
            else:
                edges_oriented[0, n], edges_oriented[1, n] = edges_oriented[1, n], edges_oriented[0, n]
    
    return edges_oriented, edge_tri_oriented

def create_triangle_to_edge_map(e2d, ed2d, edge_tri, edges, tri):
    """
    Creates a mapping from each triangle to its three edges.
    """
    elem_edges_unordered = [[] for _ in range(e2d)]
    for n in range(ed2d):
        for k in range(2):
            q = edge_tri[k, n]
            if q != -1:
                elem_edges_unordered[q].append(n)
                
    elem_edges = np.zeros((3, e2d), dtype=int)
    for elem in range(e2d):
        elnodes = tri[elem, :]
        eledges = elem_edges_unordered[elem]
        
        if len(eledges) != 3: continue
        
        for i, node_idx in enumerate(elnodes):
            for edge_idx in eledges:
                if node_idx not in edges[:, edge_idx]:
                    elem_edges[i, elem] = edge_idx
                    break
            
    return elem_edges

def calculate_dimensional_quantities(ed2d, ed2d_in, edges, edge_tri, tcenter, xcoord, ycoord, meshtype, cyclic_length, r_earth, cartesian):
    """
    Calculates geometric properties like edge lengths and distances from
    edge midpoints to triangle centers.
    """
    edge_dxdy = np.zeros((2, ed2d))
    edge_cross_dxdy = np.zeros((4, ed2d))
    rad = np.pi / 180

    if meshtype == 'm':
        edge_dxdy[0, :] = xcoord[edges[1, :]] - xcoord[edges[0, :]]
        edge_dxdy[1, :] = ycoord[edges[1, :]] - ycoord[edges[0, :]]
        
        mid_edge_x = 0.5 * (xcoord[edges[0, :]] + xcoord[edges[1, :]])
        mid_edge_y = 0.5 * (ycoord[edges[0, :]] + ycoord[edges[1, :]])
        
        edge_cross_dxdy[0, :] = tcenter[0, edge_tri[0, :]] - mid_edge_x
        edge_cross_dxdy[1, :] = tcenter[1, edge_tri[0, :]] - mid_edge_y
        
        internal_mask = np.arange(ed2d) < ed2d_in
        internal_indices = edge_tri[1, internal_mask]
        if internal_indices.size > 0:
            edge_cross_dxdy[2, internal_mask] = tcenter[0, internal_indices] - mid_edge_x[internal_mask]
            edge_cross_dxdy[3, internal_mask] = tcenter[1, internal_indices] - mid_edge_y[internal_mask]

    elif meshtype == 'r':
        for n in range(ed2d):
            ed = edges[:, n]
            def get_cyclic_diff_rad(a, b):
                diff = rad * (a - b)
                if diff > cyclic_length / 2.0: diff -= cyclic_length
                if diff < -cyclic_length / 2.0: diff += cyclic_length
                return diff

            lon_diff_rad = get_cyclic_diff_rad(xcoord[ed[1]], xcoord[ed[0]])
            lat_diff_rad = rad * (ycoord[ed[1]] - ycoord[ed[0]])
            
            a = np.array([lon_diff_rad, lat_diff_rad])
            if cartesian == False:
                a[0] *= np.cos(rad * 0.5 * np.sum(ycoord[ed]))
            edge_dxdy[:, n] = a * r_earth

            mid_lon = xcoord[ed[0]] + 0.5 * lon_diff_rad / rad
            mid_lat = 0.5 * np.sum(ycoord[ed])
            
            for k in range(2):
                tri_idx = edge_tri[k, n]
                if tri_idx != -1:
                    b_lon_diff = get_cyclic_diff_rad(tcenter[0, tri_idx], mid_lon)
                    b_lat_diff = rad * (tcenter[1, tri_idx] - mid_lat)
                    b = np.array([b_lon_diff, b_lat_diff]) * r_earth
                    if cartesian == False:
                        b[0] *= np.cos(rad * tcenter[1, tri_idx])
                    edge_cross_dxdy[2*k:2*k+2, n] = b
                    
    return edge_dxdy, edge_cross_dxdy

def calculate_laplacian_weights(e2d, ed2d_in, edge_tri, edge_dxdy, edge_cross_dxdy):
    """
    Calculates neighbor lists and weights for the Laplacian operator.
    """
    ee_pos = -np.ones((3, e2d), dtype=int)
    ee_num = np.zeros(e2d, dtype=int)
    weights = np.zeros((3, e2d))
    dxcell = np.zeros((3, e2d))
    
    for n in range(ed2d_in):
        elem1, elem2 = edge_tri[:, n]
        
        ee_pos[ee_num[elem1], elem1] = elem2
        ee_pos[ee_num[elem2], elem2] = elem1
        
        b = -edge_cross_dxdy[0:2, n] + edge_cross_dxdy[2:4, n]
        a_normal = np.array([edge_dxdy[1, n], -edge_dxdy[0, n]])
        
        dot_b = np.dot(b, b)
        weight = np.dot(a_normal, b) / dot_b if dot_b != 0 else 0.0

        weights[ee_num[elem1], elem1] = weight
        weights[ee_num[elem2], elem2] = weight
        
        dxcell[ee_num[elem1], elem1] = 0.5 * a_normal[0]
        dxcell[ee_num[elem2], elem2] = -0.5 * a_normal[0]
        
        ee_num[elem1] += 1
        ee_num[elem2] += 1
        
    return ee_pos, ee_num, weights, dxcell

def build_smoothing_and_metric(e2d, n2d, ee_num, ee_pos, elem_area, full_form, Mt=None, dxcell=None):
    # Initialize arrays, just like in the MATLAB script.
    # The shape is (4, e2d) to accommodate a diagonal (row 0) and up to 3 neighbors.
    smooth_m = np.zeros((4, e2d))
    metric = np.zeros((4, e2d))

    # Loop through each element (triangle), same as 'for j=1:e2d'
    for j in range(e2d):
        num_neighbors = ee_num[j]
        
        # Skip elements with no area to prevent division by zero
        if elem_area[j] <= 0:
            continue

        off_diagonal_value = -np.sqrt(3) / elem_area[j]
        smooth_m[1:num_neighbors + 1, j] = off_diagonal_value
        diagonal_value = -np.sum(smooth_m[1:num_neighbors + 1, j])
        smooth_m[0, j] = diagonal_value

        if full_form:
            smooth_m[0, j] += Mt[j]**2
        
            metric_values = 2 * dxcell[0:num_neighbors, j] * Mt[j] / elem_area[j]
            metric[1:num_neighbors + 1, j] = metric_values

    return smooth_m, metric

def assemble_from_intermediate(e2d, ee_num, ee_pos, smooth_m):

    # This corresponds to: nza = sum(ee_num) + e2d;
    nza = np.sum(ee_num) + e2d

    # This corresponds to: ss=zeros(...); ii=zeros(...); jj=zeros(...);
    ss = np.zeros(nza)
    ii = np.zeros(nza, dtype=int)
    jj = np.zeros(nza, dtype=int)

    # This corresponds to: nz = 0;
    nz = 0  # The counter for the current position in the ss, ii, jj arrays

    # This corresponds to the outer loop: for n=1:e2d
    for n in range(e2d):

        ss[nz] = smooth_m[0, n]  # Diagonal value is in the first row (index 0)
        
        ii[nz] = n
        jj[nz] = n
        nz += 1
        num_neighbors = ee_num[n]
        for m in range(num_neighbors):
            ss[nz] = smooth_m[m + 1, n] # Neighbor values start in row 1 of smooth_m
            ii[nz] = n
            jj[nz] = ee_pos[m, n]
            nz += 1
            
    return ss, ii, jj
