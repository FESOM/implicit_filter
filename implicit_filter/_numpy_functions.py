import numpy as np


def make_smooth(elem_area: np.ndarray, dx: np.ndarray, dy: np.ndarray, nn_num: np.ndarray,
                nn_pos: np.ndarray, tri: np.ndarray, n2d: int, e2d: int):
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


def make_smat(nn_pos: np.ndarray, nn_num: np.ndarray, smooth_m: np.ndarray, n2d: int, nza: int):
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
