from typing import Tuple
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import fori_loop, scan, cond
from functools import partial


def make_smooth(Mt: jnp.ndarray, elem_area: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, nn_num: jnp.ndarray,
                nn_pos: jnp.ndarray, tri: jnp.ndarray, n2d: int, e2d: int, full: bool = True):
    """
    Calculate the smoothness matrix and metric matrix for a given mesh.

    Parameters:
    ----------
    Mt : jnp.ndarray
        A 1D JAX NumPy array containing metric factors for each triangle.

    elem_area : jnp.ndarray
        A 1D JAX NumPy array containing the area of each triangle.

    dx : jnp.ndarray
        A 2D JAX NumPy array of shape (e2d, 3) containing x-derivatives of P1 basis functions.

    dy : jnp.ndarray
        A 2D JAX NumPy array of shape (e2d, 3) containing y-derivatives of P1 basis functions.

    nn_num : jnp.ndarray
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
    smooth_m : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing the smoothness matrix.

    metric : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing the metric matrix.
    """
    smooth_m = jnp.zeros(nn_pos.shape, dtype=jnp.float32)
    metric = jnp.zeros(nn_pos.shape, dtype=jnp.float32)
    aux = jnp.zeros((n2d,), dtype=jnp.int32)

    @jit
    def loop_body(j, carry):
        smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt = carry
        enodes = tri[j, :]

        def inner_loop_body(n, carry):
            smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area, dx, dy, Mt = carry
            row = enodes[n]
            cc = nn_num[row]

            def fill_xd(i, val):
                row, aux, nn_pos = val
                n = nn_pos[i, row]
                aux = aux.at[n].set(i)
                return row, aux, nn_pos

            row, aux, _ = fori_loop(0, cc, fill_xd, (row, aux, nn_pos))

            def update_smooth_m(m, carry):
                smooth_m, metric, aux, enodes, elem_area, dx, dy, n = carry
                col = enodes[m]
                pos = aux[col]
                tmp_x = dx[m] * dx[n]
                tmp_y = dy[n] * dy[m]
                c1 = m == n

                smooth_m = smooth_m.at[pos, row].add(cond(c1 & full,
                                                          lambda: (tmp_x + tmp_y) * elem_area + jnp.square(
                                                              Mt) * elem_area / 3.0,
                                                          lambda: (tmp_x + tmp_y) * elem_area
                                                          )
                                                     )
                metric = metric.at[pos, row].add(Mt * (dx[n] - dx[m]) * elem_area / 3.0)
                return smooth_m, metric, aux, enodes, elem_area, dx, dy, n

            smooth_m, metric, aux, _, _, _, _, _ = fori_loop(0, 3, update_smooth_m,
                                                             (smooth_m, metric, aux, enodes, elem_area, dx, dy, n))
            return smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area, dx, dy, Mt

        smooth_m, metric, aux, _, _, _, _, _, _, _ = fori_loop(0, 3, inner_loop_body, (
        smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area[j], dx[j, :], dy[j, :], Mt[j]))
        return smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt

    smooth_m, metric, _, _, _, _, _, _, _ = fori_loop(0, e2d, loop_body,
                                                      (smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt))
    return smooth_m, metric


@partial(jit, static_argnums=[3, 4])
def make_smat(nn_pos: jnp.ndarray, nn_num: jnp.ndarray, smooth_m: jnp.ndarray, n2d: int, nza: int):
    """
    Convert the smoothness matrix into a redundant sparse form (s(k), i(k), j(k)) as required by scipy.

    Parameters:
    ----------
    nn_pos : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing positions of neighboring nodes.

    nn_num : jnp.ndarray
        A 1D JAX NumPy array of shape (n2d,) containing the number of neighboring nodes for each node.

    smooth_m : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing the smoothness matrix.

    n2d : int
        The total number of nodes in the mesh.

    nza : int
        The total number of nonzero elements.

    Returns:
    -------
    ss : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the nonzero entries of the sparse matrix.

    ii : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the row indices of the nonzero entries.

    jj : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the column indices of the nonzero entries.
    """

    def helper(carry, x):
        n, m = carry
        out = (smooth_m[m, n], n, nn_pos[m, n])
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    _, tmp = scan(helper, init=(0, 0), xs=jnp.arange(nza))
    ss, ii, jj = tmp

    return ss, ii, jj


@partial(jit, static_argnums=[4, 5])
def make_smat_full(nn_pos: jnp.ndarray, nn_num: jnp.ndarray, smooth_m: jnp.ndarray, metric: jnp.ndarray, n2d: int,
                   nza: int):
    """
    Convert the output of the make_smooth function, including metric terms, into a redundant sparse form (s(k), i(k),
     j(k)) as required by scipy.

    Parameters:
    ----------
    nn_pos : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing positions of neighboring nodes.

    nn_num : jnp.ndarray
        A 1D JAX NumPy array of shape (n2d,) containing the number of neighboring nodes for each node.

    smooth_m : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing the smoothness matrix.

    metric : jnp.ndarray
        A 2D JAX NumPy array of shape (max_neighboring_nodes, n2d) containing the metric matrix.

    n2d : int
        The total number of nodes in the mesh.

    nza : int
        The total number of nonzero elements.

    Returns:
    -------
    ss : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the nonzero entries of the sparse matrix.

    ii : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the row indices of the nonzero entries.

    jj : jnp.ndarray
        A 1D JAX NumPy array of shape (nza,) containing the column indices of the nonzero entries.
    """

    def helper(carry, x):
        n, m = carry
        out = (smooth_m[m, n], n, nn_pos[m, n])
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    def helper_metric(carry, x):
        n, m = carry
        out = (metric[m, n], n, nn_pos[m, n] + n2d)
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    def helper_metric2(carry, x):
        n, m = carry
        out = (-metric[m, n], n + n2d, nn_pos[m, n])
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    def helper2(carry, x):
        n, m = carry
        out = (smooth_m[m, n], n + n2d, nn_pos[m, n] + n2d)
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    _, tmp1 = scan(helper, init=(0, 0), xs=jnp.arange(nza))
    _, tmp2 = scan(helper_metric, init=(0, 0), xs=jnp.arange(nza))
    _, tmp3 = scan(helper_metric2, init=(0, 0), xs=jnp.arange(nza))
    _, tmp4 = scan(helper2, init=(0, 0), xs=jnp.arange(nza))

    ss = jnp.concatenate((tmp1[0], tmp2[0], tmp3[0], tmp4[0]))
    ii = jnp.concatenate((tmp1[1], tmp2[1], tmp3[1], tmp4[1]))
    jj = jnp.concatenate((tmp1[2], tmp2[2], tmp3[2], tmp4[2]))

    return ss, ii, jj


def transform_vector_to_nodes(u: jnp.ndarray, v: jnp.ndarray, ne_pos: jnp.ndarray, ne_num: jnp.ndarray, n2d: int,
                                elem_area: jnp.ndarray, area: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        Project velocity components to vertices (nodes) based on the given elements' information.

        This function calculates the projected velocity components (u and v) onto the vertices (nodes) of a mesh. The
        projection is done based on the element information, including element positions, numbers, areas, and overall area.

        Parameters:
        ----------
        u : jnp.ndarray
            JAX array containing the eastward velocity component.

        v : jnp.ndarray
            JAX array containing the northward velocity component.

        ne_pos : jnp.ndarray
            JAX array of shape (ne_num, n2d) containing the positions of elements (triangles) in terms of nodes.

        ne_num : jnp.ndarray
            JAX array of shape (n2d,) containing the number of elements connected to each node.

        n2d : int
            The total number of nodes in the mesh.

        elem_area : jnp.ndarray
            JAX array of shape (e2d,) containing the areas of elements (triangles).

        area : jnp.ndarray
            JAX array of shape (n2d,) containing the areas of vertices (nodes).

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing two JAX arrays: the eastward velocity component projected to nodes (uxn) and
            the northward velocity component projected to nodes (vyn).

        Notes:
        ------
        - This function uses JAX operations to efficiently calculate the projected velocity components onto nodes.
        - The function iterates through each node and its associated elements to compute the projection.
    """
    @jit
    def calculate(ne_pos: jnp.ndarray, ne_num: int, elem_area: jnp.ndarray, area: float, vel: jnp.ndarray):
        def helper(i, val):
            out = val
            pos = ne_pos[i]
            out += vel[pos] * elem_area[pos] / 3.0
            return out

        out = fori_loop(0, ne_num, helper, (0)) / area
        return out

    uxn = vmap(lambda n: calculate(ne_pos[:, n], ne_num[n], elem_area, area[n], u))(jnp.arange(0, n2d))
    vyn = vmap(lambda n: calculate(ne_pos[:, n], ne_num[n], elem_area, area[n], v))(jnp.arange(0, n2d))

    return uxn, vyn



def transform_to_nodes(u: jnp.ndarray, ne_pos: jnp.ndarray, ne_num: jnp.ndarray, n2d: int,
                                elem_area: jnp.ndarray, area: jnp.ndarray, mask_n: jnp.ndarray) -> jnp.ndarray:
    """
        Project velocity components to vertices (nodes) based on the given elements' information.

        This function calculates the projected scalar onto the vertices (nodes) of a mesh. The
        projection is done based on the element information, including element positions, numbers, areas, and overall area.

        Parameters:
        ----------
        u : jnp.ndarray
            JAX array containing the eastward velocity component.

        ne_pos : jnp.ndarray
            JAX array of shape (ne_num, n2d) containing the indices of elements (triangles) in terms of nodes.

        ne_num : jnp.ndarray
            JAX array of shape (n2d,) containing the number of elements connected to each node.

        n2d : int
            The total number of nodes in the mesh.

        elem_area : jnp.ndarray
            JAX array of shape (e2d,) containing the areas of elements (triangles).

        area : jnp.ndarray
            JAX array of shape (n2d,) containing the areas of vertices (nodes).

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing two JAX arrays: the eastward velocity component projected to nodes (uxn) and
            the northward velocity component projected to nodes (vyn).

        Notes:
        ------
        - This function uses JAX operations to efficiently calculate the projected velocity components onto nodes.
        - The function iterates through each node and its associated elements to compute the projection.
    """
    @jit
    def calculate(ne_pos: jnp.ndarray, ne_num: int, elem_area: jnp.ndarray, area: float, mask_n: float, vel: jnp.ndarray):
        def helper(i, val):
            out = val
            pos = ne_pos[i]
            out += vel[pos] * elem_area[pos] / 3.0
            return out

        out = fori_loop(0, ne_num, helper, (0)) / area * mask_n
        return out

    # For every node, run calculate (which then goes through each element)
    uxn = vmap(lambda n: calculate(ne_pos[:, n], ne_num[n], elem_area, area[n], mask_n[n], u))(jnp.arange(0, n2d))

    return uxn




def transform_to_cells(data: jnp.ndarray, en_pos: jnp.ndarray, e2d: int,
                                elem_area: jnp.ndarray) -> jnp.ndarray:
    """
        AFW: Project data to cell centers based on the given elements' information.

        Just takes the average — So the cell centres are on the centroids...

        Parameters:
        ----------
        u : jnp.ndarray
            JAX array containing the eastward velocity component.

        en_pos : jnp.ndarray
            JAX array of shape (ne_num, n2d) containing the indices of nodes in terms of elements.

        e2d : int
            The total number of elements in the mesh.

        Returns:
        -------
        jnp.ndarray
            A JAX array: the eastward velocity component projected to cell centers (uxc).

        Notes:
        ------
        - This function uses JAX operations to efficiently calculate the projected velocity components onto cell centers.
        - The function iterates through each cell and its associated elements to compute the projection.
    """
    @jit
    def calculate(en_pos: jnp.ndarray, elem_area: jnp.ndarray, vel: jnp.ndarray):
        def helper(i, val):
            out = val
            pos = en_pos[i]
            out += vel[pos] / 3.0 * (elem_area > 0.0) # Don't interpolate over land elements (i.e. masked with elem_area = 0.0)
            return out

        out = fori_loop(0, 3, helper, (0))
        return out

    # For every cell, run calculate (which goes through each of the 3 nodes)
    uxc = vmap(lambda n: calculate(en_pos[:, n], elem_area[n], data))(jnp.arange(0, e2d))

    return uxc


def transform_vector_to_cells(u: jnp.ndarray, v: jnp.ndarray, en_pos: jnp.ndarray, e2d: int,
                                elem_area: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        AFW: Project velocity components to cell centers based on the given elements' information.

        Just takes the average — So the cell centres are on the centroids...

        Parameters:
        ----------
        u : jnp.ndarray
            JAX array containing the eastward velocity component.

        en_pos : jnp.ndarray
            JAX array of shape (ne_num, n2d) containing the indices of nodes in terms of elements.

        e2d : int
            The total number of elements in the mesh.

        Returns:
        -------
        jnp.ndarray
            A JAX array: the eastward velocity component projected to cell centers (uxc).

        Notes:
        ------
        - This function uses JAX operations to efficiently calculate the projected velocity components onto cell centers.
        - The function iterates through each cell and its associated elements to compute the projection.
    """
    @jit
    def calculate(en_pos: jnp.ndarray, elem_area: jnp.ndarray, vel: jnp.ndarray):
        def helper(i, val):
            out = val
            pos = en_pos[i]
            out += vel[pos] / 3.0 * (elem_area > 0.0)
            return out

        out = fori_loop(0, 3, helper, (0))
        return out

    # For every cell, run calculate (which goes through each of the 3 nodes)
    uxc = vmap(lambda n: calculate(en_pos[:, n], elem_area[n], u))(jnp.arange(0, e2d))
    vyc = vmap(lambda n: calculate(en_pos[:, n], elem_area[n], v))(jnp.arange(0, e2d))

    return uxc, vyc





def transform_mask_to_nodes(mask: jnp.ndarray, ne_pos: jnp.ndarray, ne_num: jnp.ndarray, n2d: int) -> jnp.ndarray:
    """
        Project _mask_ components to vertices (nodes) based on the given elements' information.

        NB: Using the normal transform function explicitly will disregard the mask !

        Parameters:
        ----------
        mask : jnp.ndarray
            JAX array containing the mask

        ne_pos : jnp.ndarray
            JAX array of shape (ne_num, n2d) containing the indices of elements (triangles) in terms of nodes.

        ne_num : jnp.ndarray
            JAX array of shape (n2d,) containing the number of elements connected to each node.

        n2d : int
            The total number of nodes in the mesh.

        elem_area : jnp.ndarray
            JAX array of shape (e2d,) containing the areas of elements (triangles).

        area : jnp.ndarray
            JAX array of shape (n2d,) containing the areas of vertices (nodes).

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing two JAX arrays: the eastward velocity component projected to nodes (uxn) and
            the northward velocity component projected to nodes (vyn).

        Notes:
        ------
        - This function uses JAX operations to efficiently calculate the projected velocity components onto nodes.
        - The function iterates through each node and its associated elements to compute the projection.
    """
    @jit
    def calculate(ne_pos: jnp.ndarray, ne_num: int, mask_ocean: jnp.ndarray):
        def helper(i, val):
            out = val
            pos = ne_pos[i]
            out += (mask_ocean[pos] > 0.5) * 1.0   # mask = 1 for ocean
            return out

        out = (fori_loop(0, ne_num, helper, (0)) > 0.5) * 1.0  # i.e. if there's at least one ocean cell that touches the node --> mask_n = 1
        return out

    # For every node, run calculate (which then goes through each element)
    uxn = vmap(lambda n: calculate(ne_pos[:, n], ne_num[n], mask))(jnp.arange(0, n2d))

    return uxn


