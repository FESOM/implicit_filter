"""
Cross-validation tests: verify that JAX functions produce the same output
as their NumPy counterparts on identical inputs.
"""
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal
from scipy.sparse import coo_matrix

from implicit_filter.utils._auxiliary import (
    make_tri,
    neighboring_triangles,
    neighbouring_nodes,
    areas,
)
from implicit_filter.utils._jax_function import (
    make_smooth as jax_make_smooth,
    make_smat as jax_make_smat,
)
from implicit_filter.utils._numpy_functions import (
    make_smooth as np_make_smooth,
    make_smat as np_make_smat,
)


def build_mesh(Lx=20, dxm=1):
    """Build a small triangular mesh for testing."""
    Ly = Lx
    dym = dxm
    xx = np.arange(0, Lx + 1, dxm)
    yy = np.arange(0, Ly + 1, dym)
    nx = len(xx)
    ny = len(yy)

    nodnum = np.arange(0, nx * ny)
    xcoord = np.zeros((nx, ny))
    ycoord = xcoord.copy()
    for i in range(nx):
        ycoord[i, :] = yy
    for i in range(ny):
        xcoord[:, i] = xx

    nodnum = np.reshape(nodnum, [ny, nx]).T
    xcoord = np.reshape(xcoord, [nx * ny])
    ycoord = np.reshape(ycoord, [nx * ny])

    tri = make_tri(nodnum, nx, ny)
    n2d = len(xcoord)
    e2d = len(tri[:, 0])

    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    mask = np.ones(e2d)
    _, elem_area, dx, dy, Mt = areas(
        n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos,
        meshtype="m", carthesian=True, cyclic_length=0, mask=mask,
    )

    return n2d, e2d, tri, nn_num, nn_pos, ne_num, ne_pos, elem_area, dx, dy, Mt


def test_make_smooth_jax_vs_numpy():
    """JAX make_smooth(full=False) must match NumPy make_smooth."""
    n2d, e2d, tri, nn_num, nn_pos, _, _, elem_area, dx, dy, Mt = build_mesh()

    # NumPy path
    smooth_np = np_make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)

    # JAX path (full=False disables metric terms, matching numpy version)
    smooth_jax, metric_jax = jax_make_smooth(
        jnp.array(Mt),
        jnp.array(elem_area),
        jnp.array(dx),
        jnp.array(dy),
        jnp.array(nn_num),
        jnp.array(nn_pos),
        jnp.array(tri),
        n2d,
        e2d,
        full=False,
    )
    smooth_jax = np.array(smooth_jax)

    assert_array_almost_equal(smooth_jax, smooth_np, decimal=5,
        err_msg="make_smooth: JAX and NumPy outputs differ")


def test_make_smat_jax_vs_numpy():
    """
    JAX and NumPy make_smat must produce the same sparse matrix
    when given the same smooth_m input.
    """
    n2d, e2d, tri, nn_num, nn_pos, _, _, elem_area, dx, dy, Mt = build_mesh()

    # Build smooth_m via NumPy (both paths should agree, but let's use
    # the NumPy smooth_m as the common input to isolate make_smat testing)
    smooth_m = np_make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)
    nza = int(np.sum(nn_num))

    # NumPy path
    ss_np, ii_np, jj_np = np_make_smat(nn_pos, nn_num, smooth_m, n2d, nza)

    # JAX path
    ss_jax, ii_jax, jj_jax = jax_make_smat(
        jnp.array(nn_pos),
        jnp.array(nn_num),
        jnp.array(smooth_m),
        n2d,
        nza,
    )
    ss_jax = np.array(ss_jax)
    ii_jax = np.array(ii_jax)
    jj_jax = np.array(jj_jax)

    # Build actual sparse matrices and compare them
    S_np = coo_matrix((ss_np, (ii_np, jj_np)), shape=(n2d, n2d)).toarray()
    S_jax = coo_matrix((ss_jax, (ii_jax, jj_jax)), shape=(n2d, n2d)).toarray()

    assert_array_almost_equal(S_jax, S_np, decimal=5,
        err_msg="make_smat: JAX and NumPy sparse matrices differ")


def test_full_pipeline_smooth_then_smat_jax_vs_numpy():
    """
    End-to-end: build smooth_m independently with each backend, then
    build the sparse matrix with each backend, and verify the resulting
    sparse matrices are identical.
    """
    n2d, e2d, tri, nn_num, nn_pos, _, _, elem_area, dx, dy, Mt = build_mesh()
    nza = int(np.sum(nn_num))

    # ---- NumPy pipeline ----
    smooth_np = np_make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)
    ss_np, ii_np, jj_np = np_make_smat(nn_pos, nn_num, smooth_np, n2d, nza)
    S_np = coo_matrix((ss_np, (ii_np, jj_np)), shape=(n2d, n2d)).toarray()

    # ---- JAX pipeline ----
    smooth_jax, _ = jax_make_smooth(
        jnp.array(Mt), jnp.array(elem_area), jnp.array(dx), jnp.array(dy),
        jnp.array(nn_num), jnp.array(nn_pos), jnp.array(tri),
        n2d, e2d, full=False,
    )
    ss_jax, ii_jax, jj_jax = jax_make_smat(
        jnp.array(nn_pos), jnp.array(nn_num), smooth_jax, n2d, nza,
    )
    S_jax = coo_matrix(
        (np.array(ss_jax), (np.array(ii_jax), np.array(jj_jax))),
        shape=(n2d, n2d),
    ).toarray()

    assert_array_almost_equal(S_jax, S_np, decimal=5,
        err_msg="Full pipeline: JAX and NumPy sparse Laplacian matrices differ")


def test_make_smooth_different_mesh_sizes():
    """Consistency holds across different mesh resolutions."""
    for Lx, dxm in [(5, 1), (10, 2), (20, 1)]:
        n2d, e2d, tri, nn_num, nn_pos, _, _, elem_area, dx, dy, Mt = build_mesh(Lx, dxm)

        smooth_np = np_make_smooth(elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d)

        smooth_jax, _ = jax_make_smooth(
            jnp.array(Mt), jnp.array(elem_area), jnp.array(dx), jnp.array(dy),
            jnp.array(nn_num), jnp.array(nn_pos), jnp.array(tri),
            n2d, e2d, full=False,
        )
        smooth_jax = np.array(smooth_jax)

        assert_array_almost_equal(smooth_jax, smooth_np, decimal=5,
            err_msg=f"make_smooth mismatch for Lx={Lx}, dxm={dxm}")
