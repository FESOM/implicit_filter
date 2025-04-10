from typing import List, Tuple

from ._utils import SolverNotConvergedError, VeryStupidIdeaError
from .jax_filter import JaxFilter
import jax.numpy as jnp
import numpy as np
import cupy
from dask import config as cfg
from dask.distributed import Client
import scipy.sparse

import pyamgx


class DotDict(dict):
    """A dictionary that supports dot notation as well as dictionary access notation."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AMGXFilter(JaxFilter):
    """
    Class uses AMGX for sparse matrix algebra and solver instead of SciPy

    This class still uses JAX for preparing auxiliary arrays on CPU
    """
    
    def init_solver(self, Smat, tol, maxiter, gpu_device=None):
        
        # Make a dict that holds mat, vec, x, resources, cfg, and solver
        # This is to avoid creating and destroying these objects every time we call the compute method
        solver_resources = DotDict()
        
        if self._full:
            # Interleave Smat and then convert to dense block sparse matrix
            Smat = self.rearrange_csr_block(Smat, self._n2d)
            solver_resources['Smat'] = Smat
        else:
            solver_resources['Smat'] = Smat
        
        # Init Resources
        solver_resources['cfg'] = self.get_config(tol, maxiter)
        if gpu_device is None:
            # Use basic
            solver_resources['resources'] = pyamgx.Resources().create_simple(solver_resources.cfg)
        else:
            solver_resources['resources'] = pyamgx.Resources().create_parallel(solver_resources.cfg, gpu_device)
        
        # Create Matrices & Vectors
        solver_resources['vec'] = pyamgx.Vector().create(solver_resources.resources, mode='dDDI')
        solver_resources['mat'] = pyamgx.Matrix().create(solver_resources.resources, mode='dDDI')
        solver_resources['x'] = pyamgx.Vector().create(solver_resources.resources, mode='dDDI')
        
        # Upload System
        if self._full:
            # Smat is a 2x2 block CSR matrix
            solver_resources.mat.upload_BSR_block(solver_resources.Smat, [self._n2d, self._n2d], [2, 2])
            solver_resources.x.set_zero(n=self._n2d, block_dim=2) #.upload(sol)
        else:
            solver_resources.mat.upload_CSR(solver_resources.Smat)
            solver_resources.x.set_zero(n=self._n2d, block_dim=1) #.upload(sol)
        
        
        
        # Create Solver
        solver_resources['solver'] = pyamgx.Solver().create(solver_resources.resources, solver_resources.cfg, mode='dDDI')
        solver_resources.solver.setup(solver_resources.mat)
        
        return solver_resources
        
    
    def finalize_solver(self, solver_resources):
        
        solver_resources.x.destroy()
        solver_resources.mat.destroy()
        solver_resources.vec.destroy()
        solver_resources.solver.destroy()
        solver_resources.resources.destroy()
        solver_resources.cfg.destroy()
    
    
    def get_config(self, tol, maxiter, method='AMG'):
        
        if method == 'AMG':
            ## Better pre-conditioned iterative solver for higher-order filters
            #  AFW 2024
            #  N.B. D^2 matrix is very non-diagonally dominant // very ill-conditioned when filter length-scale is large...
            #       Jacobi Agg AMG preconditioner helps improve convergence !  (Basic CuPy CG often does not even converge...)
            return  pyamgx.Config().create_from_dict({
                        "config_version": 2, 
                        "solver": {
                            "preconditioner": {
                                "error_scaling": 0, 
                                "algorithm": "AGGREGATION", 
                                "solver": "AMG", 
                                "smoother": "BLOCK_JACOBI", 
                                "presweeps": 4, 
                                "selector": "MULTI_PAIRWISE",   # OR "SIZE_2"
                                "coarse_solver": "NOSOLVER", 
                                "max_iters": 2,
                                "min_coarse_rows": 16, 
                                "relaxation_factor": 0.8,   # 0.8
                                "scope": "amg", 
                                "max_levels": 10, 
                                "postsweeps": 4,
                                "cycle": "V"
                            }, 
                            "use_scalar_norm": 1, 
                            "solver": "FGMRES", 
                            #"print_solve_stats": 1, 
                            "max_iters": maxiter, 
                            "monitor_residual": 1, 
                            "gmres_n_restart": 250,    # 100
                            "convergence": "ABSOLUTE", 
                            "scope": "main", 
                            "tolerance" : tol, 
                            "norm": "L2"
                        }
                    })   # Note: If get AMGX Configuration error, this usually means you haven't run pyamgx.initialize() yet...
        else:
            print('Falling back to CG Solver. The CuPy CG Solver is faster...')
            ## Basic Conjugate Gradient Method  (N.B. CuPy Implementation is Faster)
            return pyamgx.Config().create_from_dict({     
                        "config_version": 2, 
                        "determinism_flag": 1,
                            "solver": {
                                "monitor_residual": 1,
                                "solver": "CG", 
                                "convergence": "ABSOLUTE",
                                "tolerance": tol,
                                "max_iters": maxiter
                                }
                    })
            
     
        
    
    def _compute(self, n, kl, ttu, tol=1e-5, maxiter=150000, solver_resources=None, setup=True, cleanup=True, gpu_device=None) -> np.ndarray:
        
        # Replace NaNs in ttu with 0.0 (we put it back when returning)
        ttu = jnp.nan_to_num(ttu, copy=False)
        num_nans = jnp.sum(ttu)
        
        if setup:  pyamgx.initialize()
        
        if solver_resources is None:
            
            ## Construct Linear System
            Smat1 = scipy.sparse.csr_matrix((self._ss * (1.0 / np.square(kl)),
                                                (self._ii, self._jj)), shape=(self._n2d, self._n2d), dtype=np.float64)
            
            Smat = scipy.sparse.identity(self._n2d, dtype=np.float64, format='csr') + 0.5 * (Smat1 ** n) 
            
            # Initialise Matrix & Vector Resources
            solver_resources = self.init_solver(Smat, tol, maxiter, gpu_device)
        
        
        # Use perturbations (Initial Guess = 0)
        ttw = ttu - solver_resources.Smat @ ttu
        sol = np.zeros(self._n2d, dtype=np.float64) 
        
        if num_nans != 0:  # Ensure it's not just a bunch of nans...
            # Upload Transient Data:
            solver_resources.vec.upload(ttw._value.astype(np.float64))

            # Solve System
            solver_resources.solver.solve(solver_resources.vec, solver_resources.x, zero_initial_guess=True)
        
            # If solver.status is 'failed' or 'diverged', then make error message
            #print(solver.status)
            if solver_resources.solver.status == 'failed' or solver_resources.solver.status == 'diverged':
                raise SolverNotConvergedError(f"AMGX Solver failed at iteration {solver_resources.solver.iterations_number}. Residual={solver_resources.solver.get_residual}. Solver status: {solver.status}", [])

        
            # Download Solution
            solver_resources.x.download(sol)
            sol += ttu
        
        if cleanup:
            self.finalize_solver(solver_resources)
            pyamgx.finalize()
            return sol
        else:
            return sol, solver_resources
    
    
    def _compute_full(self, n, kl, ttuv, tol=1e-5, maxiter=1500000, solver_resources=None, setup=True, cleanup=True, gpu_device=None) -> np.ndarray:
        
        # Replace NaNs in ttuv with 0.0 (we put it back when returning)
        ttuv = jnp.nan_to_num(ttuv, copy=False)
        num_nans = jnp.sum(ttuv)
        
        if setup:  pyamgx.initialize()
        
        if solver_resources is None:
            
            ## Construct Linear System
            Smat1 = scipy.sparse.csr_matrix((self._ss * (1.0 / np.square(kl)),
                                                (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d), dtype=np.float64)
            
            Smat = scipy.sparse.identity(2 * self._n2d, dtype=np.float64, format='csr') + 0.5 * (Smat1 ** n)  
            
            # Initialise Matrix & Vector Resources
            solver_resources = self.init_solver(Smat, tol, maxiter, gpu_device)
        
        # Interleave ttu and ttv corresponding to 2x2 block matrix
        ttuv = self.concatenate_rhs_block(ttuv, self._n2d)
        
        # Use perturbations (Initial Guess = 0)
        ttw = ttuv - solver_resources.Smat @ ttuv
        sol = np.zeros(2 * self._n2d, dtype=np.float64) 
        
        if num_nans != 0:  # Ensure it's not just a bunch of nans...
            # Upload Transient Data:
            solver_resources.vec.upload(ttw._value.astype(np.float64), block_dim=2)

            # Solve System
            solver_resources.solver.solve(solver_resources.vec, solver_resources.x, zero_initial_guess=True)
        
            # If solver.status is 'failed' or 'diverged', then make error message
            #print(solver.status)
            if solver_resources.solver.status == 'failed' or solver_resources.solver.status == 'diverged':
                raise SolverNotConvergedError(f"AMGX Solver failed at iteration {solver_resources.solver.iterations_number}. Residual={solver_resources.solver.get_residual}. Solver status: {solver.status}", [])
        
        
            # Download Solution
            solver_resources.x.download(sol)
            sol += ttuv
        
        # Extract / Un-interleave sol and then concatenate
        sol = np.concatenate((sol[0::2], sol[1::2]))
        
        if cleanup:
            self.finalize_solver(solver_resources)
            pyamgx.finalize()
            return sol
        else:
            return sol, solver_resources
    
    
    def split_data(self, data, n_gpu):
        # Split data as evenly as possible into n_gpu chunks
        chunk_size = len(data) // n_gpu
        remainder = len(data) % n_gpu
        
        chunks = []
        start = 0
        for i in range(n_gpu):
            # Distribute the remainder elements across the first few chunks
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
        
        # Ensure there are exactly n_gpu chunks, even if some are empty
        while len(chunks) < n_gpu:
            chunks.append([])
        
        return chunks
    
    def split_data_future(self, data, n_gpu, client):
        chunk_size = len(data) // n_gpu
        worker_addresses = list(client.scheduler_info()['workers'].keys())
        
        data_futures = [None] * n_gpu
        for i, gpu_id in zip(range(0, len(data), chunk_size), range(n_gpu)):
            # Scatter each data chunk individually to the respective worker
            data_futures[gpu_id] = client.scatter(data[i:i + chunk_size], workers=[worker_addresses[gpu_id]])
        
        return data_futures
    
    def many_compute_helper(self, data_slice, gpu_id, n, kl, tol, maxiter):
        
        # If data_slice is empty...
        if len(data_slice) == 0:
            return []
        
        pyamgx.initialize()
        
        solver_resource_gpu = None
        
        results = []
        for tt in data_slice:
            tts, solver_resource_gpu = self._compute(n, kl, tt, tol, maxiter, solver_resource_gpu, setup=False, cleanup=False, gpu_device=gpu_id)
            results.append(tts)
        
        self.finalize_solver(solver_resource_gpu)
        
        pyamgx.finalize()
        
        return results
    
    
    def _many_compute(self, n, kl, data, tol=1e-5, maxiter=150000) -> List[np.ndarray]:   
        
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        cfg.set({'distributed.scheduler.worker-ttl': None})
        client = Client(n_workers=n_gpu, set_as_default=False, timeout=240)
        client.wait_for_workers(n_gpu)
        
        data_split = self.split_data(data, n_gpu)
        
        futures = []
        for j in range(n_gpu):
            futures.append(client.submit(self.many_compute_helper, data_split[j], j, n, kl, tol, maxiter, pure=False))
        
        # Gather the results from all futures
        results = client.gather(futures)
        flattened_results = [item for sublist in results for item in sublist]

        client.close()
        return flattened_results
    
    
    def rearrange_csr_block(self, csr, N): 
        
        # Extract submatrices
        A11 = csr[:N, :N].tocoo()
        A12 = csr[:N, N:].tocoo()
        A21 = csr[N:, :N].tocoo()
        A22 = csr[N:, N:].tocoo()

        # Store the new matrix data
        data = np.empty(A11.nnz + A12.nnz + A21.nnz + A22.nnz, dtype=csr.dtype)
        rows = np.empty_like(data, dtype=np.int32)
        cols = np.empty_like(data, dtype=np.int32)
        
        def interleave(A, row_offset, col_offset, start_idx):
            end_idx = start_idx + A.nnz
            rows[start_idx:end_idx] = 2 * A.row + row_offset
            cols[start_idx:end_idx] = 2 * A.col + col_offset
            data[start_idx:end_idx] = A.data
            return end_idx

        # Interleave the elements from each submatrix
        idx = 0
        idx = interleave(A11, 0, 0, idx)
        idx = interleave(A12, 0, 1, idx)
        idx = interleave(A21, 1, 0, idx)
        idx = interleave(A22, 1, 1, idx)
        
        interleaved_matrix_bsr = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(2 * N, 2 * N)).tobsr(blocksize=(2,2))

        return interleaved_matrix_bsr
    
    
    def concatenate_rhs_block(self, rhs_uv, N):
        
        b1 = rhs_uv[:N]
        b2 = rhs_uv[N:]
        
        # Fill the new RHS vector
        new_rhs = jnp.empty(2 * N, dtype=rhs_uv.dtype)
        new_rhs = new_rhs.at[0::2].set(b1)
        new_rhs = new_rhs.at[1::2].set(b2)
        
        return new_rhs
    
    
    def many_compute_full_helper(self, data_u_slice, data_v_slice, gpu_id, n, kl, tol, maxiter):
        
        # If data_slice is empty...
        if len(data_u_slice) == 0:
            return []
        
        pyamgx.initialize()
        
        solver_resource_gpu = None
        
        oux: List[np.ndarray] = list()
        ovy: List[np.ndarray] = list()
        for tt_u, tt_v in zip(data_u_slice, data_v_slice):
            
            # ttuv will be interleaved in _compute_full...
            ttuv = np.concatenate((tt_u, tt_v))
            
            ttsuv, solver_resource_gpu = self._compute_full(n, kl, ttuv, tol, maxiter, solver_resource_gpu, setup=False, cleanup=False, gpu_device=gpu_id)
            
            # ttsuv is already un-interleaved in _compute_full...
            oux.append(ttsuv[0:self._n2d])
            ovy.append(ttsuv[self._n2d:2 * self._n2d])
        
        self.finalize_solver(solver_resource_gpu)
        
        pyamgx.finalize()
        
        return oux, ovy
    
    
    def _many_compute_full(self, n, kl, data_u, data_v, tol=1e-5, maxiter=150000) -> List[np.ndarray]:   
        
        n_gpu = cupy.cuda.runtime.getDeviceCount()
        cfg.set({'distributed.scheduler.worker-ttl': None})
        client = Client(n_workers=n_gpu, set_as_default=False, timeout=240)
        client.wait_for_workers(n_gpu)
        
        data_u_split = self.split_data(data_u, n_gpu)
        data_v_split = self.split_data(data_v, n_gpu)
        
        futures = []
        for j in range(n_gpu):
            futures.append(client.submit(self.many_compute_full_helper, data_u_split[j], data_v_split[j], j, n, kl, tol, maxiter, pure=False))
        
        # Gather the results from all futures
        results = client.gather(futures)
        oux = [item for sublist in results for item in sublist[0]]
        ovy = [item for sublist in results for item in sublist[1]]

        client.close()
        return oux, ovy










