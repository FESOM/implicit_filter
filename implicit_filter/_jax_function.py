import jax.numpy as jnp
from jax import jit
import jax
from jax.lax import fori_loop, scan, cond
from functools import partial


def make_smooth(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, full=True):
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
                                                            lambda : (tmp_x + tmp_y) * elem_area + jnp.square(Mt) * elem_area / 3.0,
                                                            lambda : (tmp_x + tmp_y) * elem_area
                                                            )
                                                     )
                metric = metric.at[pos, row].add(Mt * (dx[n] - dx[m]) * elem_area / 3.0)
                return smooth_m, metric, aux, enodes, elem_area, dx, dy, n


            smooth_m, metric, aux, _, _, _, _, _ = fori_loop(0, 3, update_smooth_m, (smooth_m, metric, aux, enodes, elem_area, dx, dy, n))
            return smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area, dx, dy, Mt


        smooth_m, metric, aux, _, _, _, _, _, _, _ = fori_loop(0, 3, inner_loop_body, (smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area[j], dx[j, :], dy[j, :], Mt[j]))
        return smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt

    smooth_m, metric, _, _, _, _, _, _, _ = fori_loop(0, e2d, loop_body, (smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt))
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
