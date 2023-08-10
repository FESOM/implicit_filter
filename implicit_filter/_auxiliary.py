import math
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


def neighbouring_nodes(n2d: int, tri: np.ndarray, ne_num: np.ndarray, ne_pos: np.ndarray):
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



def areas(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, ne_num: np.ndarray,
          ne_pos: np.ndarray, meshtype: str, carthesian: bool, cyclic_length):
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
        A 1D NumPy array containing the x-coordinates of nodes in the mesh.

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

    if meshtype == 'm':
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
            elem_area[n] = 0.5 * abs(d)

    elif meshtype == 'r':
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
            elem_area[n] = 0.5 * abs(d)

        if carthesian:
            Mt = np.zeros([e2d])
        else:
            Mt = (np.sin(rad * np.sum(ycoord[tri], axis=1) / 3.0) / Mt) / r_earth

    # Calculate scalar cell (cluster) area for each node.
    area = np.zeros([n2d])
    for n in range(n2d):
        area[n] = np.sum(elem_area[ne_pos[0:ne_num[n], n]]) / 3.0

    return area, elem_area, dx, dy, Mt
