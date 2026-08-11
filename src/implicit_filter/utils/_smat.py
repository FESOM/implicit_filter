""" 
Helper file to define the sparse matrix that is used in the _batch_compute 
functions of all filters. 
"""


from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import ops
import numpy as np
import scipy.sparse as sp
import jax.scipy.sparse.linalg as jspsl



# ---------------------------------------------------------------------------
# Sparse matrix container (PyTree)
# ---------------------------------------------------------------------------

@dataclass
class SmatData:
    """
    Define a PyTree class to hold the sparse matrix data in CSR format, 
    along with the diagonal and row IDs.
    
    """
    data: jnp.ndarray
    indices: jnp.ndarray
    indptr: jnp.ndarray
    diag: jnp.ndarray
    row_ids: jnp.ndarray   
    shape: tuple

# Register pytree
jax.tree_util.register_pytree_node(
    SmatData,
    lambda s: ([s.data, s.indices, s.indptr, s.diag, s.row_ids], s.shape),
    lambda shape, leaves: SmatData(
        leaves[0], leaves[1], leaves[2], leaves[3], leaves[4], shape
    ),
)

# ---------------------------------------------------------------------------
# Build sparse matrix 
# ---------------------------------------------------------------------------

def build_smat(
    X,
    kl: float | np.ndarray,
    is_elem: bool = False,
    ):
    """
    Build the raw scaled stencil matrix for the given filter and filter wavelength.

    Assembly runs on the host with scipy (mirroring how ``prepare()`` builds
    the mesh operator), and returns a scipy sparse matrix rather than
    :class:`SmatData` so callers (e.g. ``Filter._batch_compute``) can combine
    it into stage matrices (``I + c1*Smat + c2*Smat**2``) with ordinary
    sparse arithmetic before converting each stage to :class:`SmatData` via
    :func:`to_smat_data`, which moves it onto the JAX device.

    Parameters
    ----------
    X : Filter
        The filter object (``TriangularFilter`` or ``LatLonFilter``)
        containing the grid information and filter coefficients.
    kl : float | np.ndarray
        Filter wavelength in spatial units.
        Can be a float (applied to entire mesh) or an array with scales for each node or element.
        Size of the array must match the size of the input data.
    is_elem : bool, optional
        If True, the filter is applied to elements; if False, it is applied to nodes.
        Default is False (node-based filtering).
        Triggered by the _batch_compute function in the filters.

    Returns
    -------
    scipy.sparse.csr_matrix
        The scaled stencil matrix, signed to match ``X``'s own ``_compute``
        convention (see ``X._STENCIL_SIGN``: +1 for TriangularFilter,
        -1 for LatLonFilter).
    """
    n_size = int(X._n2d if not is_elem else X._e2d)
    ss = np.asarray(X._ss_e if is_elem else X._ss)
    ii = np.asarray(X._ii_e if is_elem else X._ii)
    jj = np.asarray(X._jj_e if is_elem else X._jj)

    kl = np.full(n_size, kl) if isinstance(kl, (float, int)) else np.asarray(kl)

    Smat = sp.csc_matrix((ss, (ii, jj)), shape=(n_size, n_size))

    sign = getattr(X, "_STENCIL_SIGN", 1.0)
    scaling_vector = sign / np.square(kl)
    nnz_per_column = np.diff(Smat.indptr)
    multipliers = np.repeat(scaling_vector, nnz_per_column)
    Smat.data *= multipliers

    return Smat.tocsr()


def to_smat_data(Smat_scipy, convers=jnp.array) -> SmatData:
    """Convert a scipy/cupy CSR (or CSC) sparse matrix to :class:`SmatData`."""
    csr = Smat_scipy.tocsr()
    diag = np.array(csr.diagonal())
    row_ids = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
    return SmatData(
        data=convers(csr.data),
        indices=convers(csr.indices),
        indptr=convers(csr.indptr),
        diag=convers(diag),
        row_ids=convers(row_ids),
        shape=csr.shape,
    )

def _matvec(
    smat: SmatData, 
    x: jnp.ndarray
    ) -> jnp.ndarray:
    """
    Perform the matrix-vector product of the sparse matrix represented by 
    SmatData with a dense vector x (data). This is used in the _batch_compute 
    functions of the filters to apply the filter operation.

    """
    products = smat.data * x[smat.indices]
    out = ops.segment_sum(
        products,
        smat.row_ids,
        num_segments=smat.shape[0],
    )
    return out 

def _cg_one(
    smat: SmatData,
    ttu: jnp.ndarray,
    tol = 1e-6,
    maxiter = 150_000,
    ) -> jnp.ndarray:
    """
    """

    def matvec(x):
        return _matvec(smat, x)

    ttw = ttu - matvec(ttu)

    pre = lambda x: x / smat.diag  # Jacobi preconditioner

    sol, info = jspsl.cg(
        matvec,
        ttw,
        tol=tol,
        maxiter=maxiter,
        M=pre,
    )

    return sol + ttu, info

