""" 
Helper file to define the sparse matrix that is used in the _batch_compute 
functions of all filters. 
"""


from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import ops
import numpy as np
import jax.scipy.sparse.linalg as jspsl
from implicit_filter.filter import Filter



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
    X : Filter, 
    kl:float | np.ndarray,
    is_elem: bool = False
    ) -> SmatData:
    """
    Build the sparse matrix in CSR format for the given filter and filter wavelength.

    Parameters
    ----------
    X : Filter
        The filter object containing the grid information and filter coefficients.
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
    SmatData
        A SmatData object containing the sparse matrix data in CSR format, 
        along with the diagonal and row IDs.
    """

    n_size = X._n2d if not is_elem else X._e2d
    ss = X._ss_e if is_elem else X._ss
    ii = X._ii_e if is_elem else X._ii
    jj = X._jj_e if is_elem else X._jj

    if isinstance(kl, float):
        kl = np.full(n_size, kl)

    Smat = X.csc_matrix(X.convers(ss),
                        X.convers(ii),
                        X.convers(jj),
                        shape=(n_size, n_size))
    
    scaling_vector = 1.0 / np.square(kl)
    nnz_per_column = np.diff(X.tonumpy(Smat.indptr))
    multipliers = np.repeat(scaling_vector, nnz_per_column)
    Smat.data *= X.convers(multipliers)

    Smat_csr = Smat.tocsr()
    diag = X.tonumpy(Smat_csr.diagonal())
    row_ids = np.repeat(
        np.arange(n_size), 
        np.diff(X.tonumpy(Smat_csr.indptr))
    )

    return SmatData(
        data = X.convers(Smat_csr.data),
        indices = X.convers(Smat_csr.indices),
        indptr = X.convers(Smat_csr.indptr),
        diag = X.convers(diag),
        row_ids = X.convers(row_ids),
        shape = (n_size, n_size)
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

