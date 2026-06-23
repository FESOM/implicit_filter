import pytest
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

# Ensure 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

def test_sparse_spmv():
    """
    Test that the JAX scatter_add sparse matrix-vector multiplication 
    matches SciPy's sparse CSR multiplication exactly.
    """
    n = 100
    # Create random sparse data
    nnz = 500
    ii = np.random.randint(0, n, nnz)
    jj = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz)
    
    # Random vector
    x = np.random.randn(n)
    
    # 1. Compute with SciPy (our baseline reference)
    scipy_mat = sp.coo_matrix((data, (ii, jj)), shape=(n, n)).tocsr()
    expected_result = scipy_mat @ x
    
    # 2. Compute with our JAX scatter_add implementation
    jnp_x = jnp.array(x)
    jnp_data = jnp.array(data)
    jnp_ii = jnp.array(ii)
    jnp_jj = jnp.array(jj)
    
    jax_result = jnp.zeros_like(jnp_x).at[jnp_ii].add(jnp_data * jnp_x[jnp_jj])
    
    # Check if results are equivalent
    np.testing.assert_allclose(jax_result, expected_result, rtol=1e-12, atol=1e-12)

def test_sparse_spmv_repeated():
    """
    Test repeated application (power of matrix) matching SciPy.
    """
    n = 50
    # Create random symmetric positive definite-like sparse data
    nnz = 200
    ii = np.random.randint(0, n, nnz)
    jj = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz)
    
    # Symmetrize
    ii_sym = np.concatenate([ii, jj])
    jj_sym = np.concatenate([jj, ii])
    data_sym = np.concatenate([data, data])
    
    x = np.random.randn(n)
    
    scipy_mat = sp.coo_matrix((data_sym, (ii_sym, jj_sym)), shape=(n, n)).tocsr()
    expected_result = scipy_mat @ (scipy_mat @ x)
    
    jnp_x = jnp.array(x)
    jnp_data = jnp.array(data_sym)
    jnp_ii = jnp.array(ii_sym)
    jnp_jj = jnp.array(jj_sym)
    
    # 2 passes
    y = jnp.zeros_like(jnp_x).at[jnp_ii].add(jnp_data * jnp_x[jnp_jj])
    jax_result = jnp.zeros_like(y).at[jnp_ii].add(jnp_data * y[jnp_jj])
    
    np.testing.assert_allclose(jax_result, expected_result, rtol=1e-12, atol=1e-12)
