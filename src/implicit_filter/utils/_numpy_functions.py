from typing import Tuple

import numpy as np
import pandas as pd


def make_smooth(
    elem_area: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    nn_num: np.ndarray,
    nn_pos: np.ndarray,
    tri: np.ndarray,
    n2d: int,
    e2d: int,
):
    """
    Calculate the smoothness matrix and metric matrix for a given mesh.
    It does not support metric terms

    Parameters:
    ----------
    elem_area : np.ndarray
        A 1D NumPy array containing the area of each triangle.

    dx : np.ndarray
        A 2D NumPy array of shape (e2d, 3) containing x-derivatives of P1 basis functions.

    dy : np.ndarray
        A 2D NumPy array of shape (e2d, 3) containing y-derivatives of P1 basis functions.

    nn_num : np.ndarray
        A 1D JAX NumPy array of shape (n2d,) containing the number of neighboring nodes for each node.

    nn_pos : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing positions of neighboring nodes.

    tri : jnp.ndarray
        A 2D JAX NumPy array representing the connectivity of triangles in the mesh.

    n2d : int
        The total number of nodes in the mesh.

    e2d : int
        The total number of triangles (elements) in the mesh.

    full : bool, optional
        A flag indicating whether to use the 'full' calculation including metric factors (True) or not (False).
        Default is True.

    Returns:
    -------
    smooth_m : np.ndarray
        A 2D NumPy array of shape (max_neighboring_nodes, n2d) containing the smoothness matrix.

    """

    smooth_m = np.zeros(nn_pos.shape)  # Place for non - zero entries
    aux = np.zeros([n2d], dtype=int)  # auxiliary array
    for j in range(e2d):
        enodes = tri[j, :]  # vertices of triangle
        hh = np.sqrt(2 * elem_area[j])
        # This expression corresponds to the specific mesh above obtained
        # by splitting quads. A slightly different expression is needed
        # for equilateral triangles.

        for n in range(3):  # Each contributes to rows
            row = enodes[n]  # and columns
            cc = nn_num[row]  # num of neighbors
            tmp = nn_pos[0:cc, row]
            aux[tmp] = np.arange(0, cc)  # Map their order
            for m in range(3):
                col = enodes[m]
                pos = aux[col]  # Position of column among neighbors
                tmp_x = dx[j, m] * dx[j, n]
                tmp_y = dy[j, n] * dy[j, m]
                smooth_m[pos, row] += (tmp_x + tmp_y) * elem_area[j]

    return smooth_m


def make_smat(
    nn_pos: np.ndarray, nn_num: np.ndarray, smooth_m: np.ndarray, n2d: int, nza: int
):
    """
    Convert the smoothness matrix into a redundant sparse form (s(k), i(k), j(k)) as required by scipy.

    Parameters:
    ----------
    nn_pos : np.ndarray
        A 2D NumPy array of shape (max_neighboring_nodes, n2d) containing positions of neighboring nodes.

    nn_num : np.ndarray
        A 1D NumPy array of shape (n2d,) containing the number of neighboring nodes for each node.

    smooth_m : np.ndarray
        A 2D NumPy array of shape (max_neighboring_nodes, n2d) containing the smoothness matrix.

    n2d : int
        The total number of nodes in the mesh.

    nza : int
        The total number of nonzero elements.

    Returns:
    -------
    ss : np.ndarray
        A 1D NumPy array of shape (nza,) containing the nonzero entries of the sparse matrix.

    ii : np.ndarray
        A 1D NumPy array of shape (nza,) containing the row indices of the nonzero entries.

    jj : np.ndarray
        A 1D NumPy array of shape (nza,) containing the column indices of the nonzero entries.
    """
    nza = np.sum(nn_num)  # The number of nonzero elements:
    ss = np.zeros([nza])  # Place for nonzero entries
    ii = np.zeros([nza])  # Place for their rows
    jj = np.zeros([nza])  # Place for their columns
    nz = 0
    for n in range(n2d):
        for m in range(nn_num[n]):
            ss[nz] = smooth_m[m, n]
            ii[nz] = n
            jj[nz] = nn_pos[m, n]
            nz += 1

    return ss, ii, jj


def convert_to_tcells(
    e2d: int, ee_pos: np.ndarray, ux: np.ndarray, vy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros((4, e2d))  # N, W, E, S
    v[0, :] = np.reshape(ux, e2d)  # West
    v[1, :] = np.reshape(vy, e2d)  # North

    for i in range(e2d):
        if ee_pos[3, i] != i:
            v[3, i] = v[1, ee_pos[3, i]]

        if ee_pos[2, i] != i:
            v[2, i] = v[0, ee_pos[2, i]]

    ttu = v[0, :] + v[2, :]
    ttv = v[1, :] + v[3, :]

    return np.reshape(ttu, ux.shape), np.reshape(ttv, vy.shape)


def calculate_global_nemo_neighbourhood(
    e2d: int, nx: int, ny: int, north_adj: pd.Series
) -> tuple[np.ndarray, int]:
    """
    Calculate neighbourhood of each cell in NEMO global mesh

    Parameters:
    ----------
    e2d : int
        The total number of cells (elements) in the mesh.

    nx : int
        Number of cells in X direction.

    ny : int
        Number of cells in Y direction.

    north_adj : pd.Series
        Pandas Series containing the indices of adjacent points at northern border.

    Returns:
    -------
    ee_pos : np.ndarray
        2D NumPy array of shape (4, e2d) containing indexes of neighboring cells.

    nza : int
        The total number of unique elements in ee_pos.
    """

    ee_pos = np.zeros((4, e2d), dtype=np.int32)
    # ids = set()
    # Initialize nza
    nza = 0

    yc = ny  # Numer of cells in y axis
    xc = nx  # Number of cells in x axis

    # Fill ee_pos, arrangement is W;N;E;S
    ee_pos[:, 0] = np.array([yc * (xc - 1), 1, yc, 0])  # Corner
    # print(f"x: {0} y: {0} ni: {0}")
    # ids.add(0)
    nza += 3
    for m in range(1, yc - 1):
        ee_pos[:, m] = [yc * (xc - 1) + m, m + 1, m + yc, m - 1]  # Left border
        # ids.add(m)
        # print(f"x: {0} y: {m} ni: {m}")
        nza += 4
    # print("Left")
    m = yc - 1
    # print(f"Northern neighbour of {1} is {north_adj[1]}")
    ee_pos[:, m] = [xc * yc - 1, yc * north_adj[1] - 1, m + yc, m - 1]  # Second corner
    # ids.add(m)
    # print(f"x: {0} y: {m} ni: {ni}")
    nza += 4

    for n in range(1, xc - 1):  # Center
        # print("Bottom border")
        no = yc * n
        ni = no
        ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {0} ni: {ni}")
        nza += 3
        # print("Inner center")
        for m in range(1, yc - 1):
            ni = no + m
            ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni - 1]
            # if not ni in ids: ids.add(ni)
            # print(f"x: {n} y: {m} ni: {ni}")
            nza += 4

        # print("Top border")
        ni = no + (yc - 1)
        # print(f"Northern neighbour of {n + 1} is {north_adj[n + 1]}")
        ee_pos[:, ni] = [ni - yc, north_adj[n + 1] * yc - 1, ni + yc, ni - 1]

        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {yc-1} ni: {ni}")
        nza += 4

    no = yc * (xc - 1)
    ni = no
    ee_pos[:, ni] = [ni - yc, ni + 1, 0, ni]  # Third corner
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {0} ni: {ni}")
    nza += 3
    # print("Right")
    for m in range(1, yc - 1):  # Right border
        ni = no + m
        ee_pos[:, ni] = [ni - yc, ni + 1, m, ni - 1]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {xc-1} y: {m} ni: {ni}")
        nza += 4

    ni = no + (yc - 1)
    # print(f"Northern neighbour of {xc} is {north_adj[xc]}")
    ee_pos[:, ni] = [ni - yc, north_adj[xc] * yc - 1, yc - 1, ni - 1]  # Fourth border
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {yc-1} ni: {ni}")
    nza += 4

    # Final adjustment for nza
    nza += e2d

    return ee_pos, nza


def calculate_global_regular_neighbourhood(
    e2d: int, nx: int, ny: int
) -> Tuple[np.ndarray, int]:
    """
    Calculate the neighbourhood of each cell in regular global mesh.
    The Northern border is ignored

    Parameters:
    ----------
    e2d : int
        The total number of cells (elements) in the mesh.

    nx : int
        Number of cells in X direction.

    ny : int
        Number of cells in Y direction.

    Returns:
    -------
    ee_pos : np.ndarray
        2D NumPy array of shape (4, e2d) containing indexes of neighboring cells.

    nza : int
        The total number of unique elements in ee_pos.
    """

    ee_pos = np.zeros((4, e2d), dtype=np.int32)
    # ids = set()
    # Initialize nza
    nza = 0

    yc = ny  # Numer of cells in y axis
    xc = nx  # Number of cells in x axis

    # Fill ee_pos, arrangement is W;N;E;S
    ee_pos[:, 0] = np.array([yc * (xc - 1), 1, yc, 0])  # Corner
    # print(f"x: {0} y: {0} ni: {0}")
    # ids.add(0)
    nza += 3
    for m in range(1, yc - 1):
        ee_pos[:, m] = [yc * (xc - 1) + m, m + 1, m + yc, m - 1]  # Left border
        # ids.add(m)
        # print(f"x: {0} y: {m} ni: {m}")
        nza += 4
    # print("Left")
    m = yc - 1
    # print(f"Northern neighbour of {1} is {north_adj[1]}")
    ee_pos[:, m] = [xc * yc - 1, m, m + yc, m - 1]  # Second corner
    # ids.add(m)
    # print(f"x: {0} y: {m} ni: {ni}")
    nza += 3

    for n in range(1, xc - 1):  # Center
        # print("Bottom border")
        no = yc * n
        ni = no
        ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {0} ni: {ni}")
        nza += 3
        # print("Inner center")
        for m in range(1, yc - 1):
            ni = no + m
            ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni - 1]
            # if not ni in ids: ids.add(ni)
            # print(f"x: {n} y: {m} ni: {ni}")
            nza += 4

        # print("Top border")
        ni = no + (yc - 1)
        # print(f"Northern neighbour of {n + 1} is {north_adj[n + 1]}")
        ee_pos[:, ni] = [ni - yc, ni, ni + yc, ni - 1]

        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {yc-1} ni: {ni}")
        nza += 3

    no = yc * (xc - 1)
    ni = no
    ee_pos[:, ni] = [ni - yc, ni + 1, 0, ni]  # Third corner
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {0} ni: {ni}")
    nza += 3
    # print("Right")
    for m in range(1, yc - 1):  # Right border
        ni = no + m
        ee_pos[:, ni] = [ni - yc, ni + 1, m, ni - 1]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {xc-1} y: {m} ni: {ni}")
        nza += 4

    ni = no + (yc - 1)
    # print(f"Northern neighbour of {xc} is {north_adj[xc]}")
    ee_pos[:, ni] = [ni - yc, ni, yc - 1, ni - 1]  # Fourth border
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {yc-1} ni: {ni}")
    nza += 3

    # Final adjustment for nza
    nza += e2d

    return ee_pos, nza


def calculate_local_regular_neighbourhood(
    e2d: int, nx: int, ny: int
) -> Tuple[np.ndarray, int]:
    """
    Calculate the neighbourhood of each cell in regular local mesh.
    All periodic neighbours are ignored

    Parameters:
    ----------
    e2d : int
        The total number of cells (elements) in the mesh.

    nx : int
        Number of cells in X direction.

    ny : int
        Number of cells in Y direction.

    Returns:
    -------
    ee_pos : np.ndarray
        2D NumPy array of shape (4, e2d) containing indexes of neighboring cells.

    nza : int
        The total number of unique elements in ee_pos.
    """

    ee_pos = np.zeros((4, e2d), dtype=np.int32)
    # ids = set()
    # Initialize nza
    nza = 0

    yc = ny  # Numer of cells in y axis
    xc = nx  # Number of cells in x axis

    # Fill ee_pos, arrangement is W;N;E;S
    ee_pos[:, 0] = np.array([0, 1, yc, 0])  # Corner
    # print(f"x: {0} y: {0} ni: {0}")
    # ids.add(0)
    nza += 2
    for m in range(1, yc - 1):
        ee_pos[:, m] = [m, m + 1, m + yc, m - 1]  # Left border
        # ids.add(m)
        # print(f"x: {0} y: {m} ni: {m}")
        nza += 3
    # print("Left")
    m = yc - 1
    # print(f"Northern neighbour of {1} is {north_adj[1]}")
    ee_pos[:, m] = [m, m, m + yc, m - 1]  # Second corner
    # ids.add(m)
    # print(f"x: {0} y: {m} ni: {ni}")
    nza += 2

    for n in range(1, xc - 1):  # Center
        # print("Bottom border")
        no = yc * n
        ni = no
        ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {0} ni: {ni}")
        nza += 3
        # print("Inner center")
        for m in range(1, yc - 1):
            ni = no + m
            ee_pos[:, ni] = [ni - yc, ni + 1, ni + yc, ni - 1]
            # if not ni in ids: ids.add(ni)
            # print(f"x: {n} y: {m} ni: {ni}")
            nza += 4

        # print("Top border")
        ni = no + (yc - 1)
        # print(f"Northern neighbour of {n + 1} is {north_adj[n + 1]}")
        ee_pos[:, ni] = [ni - yc, ni, ni + yc, ni - 1]

        # if not ni in ids: ids.add(ni)
        # print(f"x: {n} y: {yc-1} ni: {ni}")
        nza += 3

    no = yc * (xc - 1)
    ni = no
    ee_pos[:, ni] = [ni - yc, ni + 1, ni, ni]  # Third corner
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {0} ni: {ni}")
    nza += 2
    # print("Right")
    for m in range(1, yc - 1):  # Right border
        ni = no + m
        ee_pos[:, ni] = [ni - yc, ni + 1, ni, ni - 1]
        # if not ni in ids: ids.add(ni)
        # print(f"x: {xc-1} y: {m} ni: {ni}")
        nza += 3

    ni = no + (yc - 1)
    # print(f"Northern neighbour of {xc} is {north_adj[xc]}")
    ee_pos[:, ni] = [ni - yc, ni, ni, ni - 1]  # Fourth border
    # if not ni in ids: ids.add(ni)
    # print(f"x: {xc-1} y: {yc-1} ni: {ni}")
    nza += 2

    # Final adjustment for nza
    nza += e2d

    return ee_pos, nza
