from typing import List, Tuple

from ._utils import SolverNotConvergedError
from .jax_filter import JaxFilter
import jax.numpy as jnp
import numpy as np
import cupy
import concurrent
import scipy.sparse

import pyamgx


class AMGXFilter(JaxFilter):
    """
    Class uses AMGX for sparse matrix algebra and solver instead of SciPy

    This class still uses JAX for preparing auxiliary arrays on CPU
    """
    
    def init_solver(self):
        pyamgx.initialize()
    
    def finalize_solver(self):
        pyamgx.finalize()
    
    def set_config(self, config):
        self._config = config
    
    def _compute(self, n, kl, ttu, tol=1e-5, maxiter=150000) -> np.ndarray:
        
        pyamgx.initialize()
        
        # cfg = pyamgx.Config()
        # cfg.create_from_file('/home/b/b382615/opt/AMGX/src/configs/CG_DILU.json')
        
        # cfg = pyamgx.Config().create_from_dict({   # WORKS !  (1st Order)
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "monitor_residual": 1,
        #             "solver": "CG", 
        #             "convergence": "ABSOLUTE",
        #             "tolerance": tol,
        #             "max_iters": maxiter
        #             }
        # })


        
        # cfg = pyamgx.Config().create_from_dict({    ## GMRES + AMG   # WORKS with 2nd Order (?)
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "preconditioner": {
        #                 "error_scaling": 0, 
        #                 "print_grid_stats": 1, 
        #                 "algorithm": "AGGREGATION", 
        #                 "solver": "AMG", 
        #                 "smoother": "BLOCK_JACOBI", 
        #                 "presweeps": 0, 
        #                 "selector": "SIZE_2", 
        #                 "coarse_solver": "NOSOLVER", 
        #                 "max_iters": 1, 
        #                 "min_coarse_rows": 32, 
        #                 "relaxation_factor": 0.75, 
        #                 "scope": "amg", 
        #                 "max_levels": 100, 
        #                 "postsweeps": 3, 
        #                 "cycle": "V"
        #             }, 
        #             "use_scalar_norm": 1, 
        #             "solver": "FGMRES", 
        #             "print_solve_stats": 1, 
        #             "obtain_timings": 1, 
        #             "max_iters": 1000, 
        #             "monitor_residual": 1, 
        #             "gmres_n_restart": 32, 
        #             "convergence": "RELATIVE_INI", 
        #             "scope": "main", 
        #             "tolerance" : 1e-06, 
        #             "norm": "L2"
        #         }
        # })
        
        
        # cfg = pyamgx.Config().create_from_dict({     ## Custom // Perplexity -- Doesn't work....
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "preconditioner": {
        #                 "algorithm": "CLASSICAL",
        #                 "selector": "HMIS",
        #                 "solver": "AMG", 
        #                 "smoother": {
        #                     "solver": "MULTICOLOR_DILU", 
        #                     "scope": "level"
        #                 }, 
        #                 "presweeps": 1, 
        #                 "cycle": "V", 
        #                 "postsweeps": 1,
        #                 "coarse_solver": "DENSE_LU_SOLVER",
        #                 "scope": "amg",
        #             }, 
        #             "solver": "PCG", 
        #             "max_iters": 1000, 
        #             "monitor_residual": 1, 
        #             "convergence": "ABSOLUTE", 
        #             "scope": "main", 
        #             "tolerance" : 1e-06, 
        #             "norm": "L2"
        #         }
        # })
        
        
        
        # cfg = pyamgx.Config().create_from_dict({   # WORKS !  (1st Order)
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "monitor_residual": 1,
        #             "print_solve_stats": 1,
        #             "solver": "CG", 
        #             "convergence": "ABSOLUTE",
        #             "tolerance": 1e-5,
        #             "max_iters": 1000
        #             }
        # })
        
        # cfg = pyamgx.Config().create_from_dict({  
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "preconditioner": {
        #                 "scope": "amg", 
        #                 "solver": "NOSOLVER"
        #             }, 
        #             "solver": "PCG", 
        #             "max_iters": 500, 
        #             "print_solve_stats": 1,
        #             "monitor_residual": 1, 
        #             "scope": "main", 
        #             "tolerance" : 1e-05, 
        #             "convergence": "ABSOLUTE",
        #         }
        # })
        
        
        # cfg = pyamgx.Config().create_from_dict({  
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "preconditioner": {
        #                 "solver": "AMG", 
        #                 "smoother": {
        #                     "scope": "jacobi", 
        #                     "solver": "BLOCK_JACOBI", 
        #                     "monitor_residual": 0, 
        #                 }, 
        #                 "presweeps": 1, 
        #                 "interpolator": "D2",
        #                 "max_iters": 1, 
        #                 "monitor_residual": 0, 
        #                 "scope": "amg", 
        #                 "max_levels": 100, 
        #                 "cycle": "V", 
        #                 "postsweeps": 1
        #             }, 
        #             "solver": "PBICGSTAB", 
        #             "max_iters": 200, 
        #             "print_solve_stats": 1,
        #             "monitor_residual": 1, 
        #             "scope": "main", 
        #             "tolerance" : 1e-06, 
        #             "convergence": "ABSOLUTE",
        #         }
        # })
        
        # cfg = pyamgx.Config().create_from_dict({  
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "solver": "GMRES", 
        #             "print_solve_stats": 1, 
        #             "preconditioner": {
        #                 "interpolator": "D2", 
        #                 "solver": "AMG", 
        #                 "smoother": "JACOBI_L1", 
        #                 "presweeps": 2, 
        #                 "selector": "PMIS", 
        #                 "coarsest_sweeps": 2, 
        #                 "coarse_solver": "NOSOLVER", 
        #                 "max_iters": 1, 
        #                 "interp_max_elements": 4, 
        #                 "min_coarse_rows": 2, 
        #                 "scope": "amg_solver", 
        #                 "max_levels": 24, 
        #                 "cycle": "V", 
        #                 "postsweeps": 2
        #             }, 
        #             "max_iters": 200, 
        #             "monitor_residual": 1, 
        #             "gmres_n_restart": 10, 
        #             "convergence": "ABSOLUTE", 
        #             "tolerance" : 1e-06, 
        #             "norm": "L2"
        #     }
        # })
        
        
        # cfg = pyamgx.Config().create_from_dict({     # Consistent but slow convergence... for O2
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "preconditioner": {
        #                 "scope": "amg", 
        #                 "solver": "NOSOLVER"
        #             }, 
        #             "use_scalar_norm": 1, 
        #             "solver": "GMRES", 
        #             "print_solve_stats": 1, 
        #             "max_iters": 200, 
        #             "monitor_residual": 1, 
        #             "convergence": "ABSOLUTE", 
        #             "tolerance" : 1e-06, 
        #     }
        # })
        
        # cfg = pyamgx.Config().create_from_dict({  
        #     "config_version": 2, 
        #     "determinism_flag": 1,
        #         "solver": {
        #             "solver": "GMRES", 
        #             "obtain_timings": 1, 
        #             "preconditioner": {
        #                 "interpolator": "D2", 
        #                 "solver": "AMG", 
        #                 "smoother": "JACOBI_L1", 
        #                 "presweeps": 2, 
        #                 "selector": "PMIS", 
        #                 "coarsest_sweeps": 2, 
        #                 "coarse_solver": "NOSOLVER", 
        #                 "max_iters": 1, 
        #                 "interp_max_elements": 4, 
        #                 "min_coarse_rows": 2, 
        #                 "scope": "amg_solver", 
        #                 "max_levels": 24, 
        #                 "cycle": "V", 
        #                 "postsweeps": 2
        #             }, 
        #             "max_iters": 100, 
        #             "monitor_residual": 1, 
        #             "gmres_n_restart": 10, 
        #             "convergence": "RELATIVE_INI", 
        #             "tolerance" : 1e-06, 
        #             "norm": "L2"
        #     }
        # })
        
        
        cfg = pyamgx.Config().create_from_dict({   # WORKS !  (1st Order)
            "config_version": 2, 
            "determinism_flag": 1,
                "solver": {
                    "preconditioner": {
                        "scope": "precond", 
                        "solver": "MULTICOLOR_DILU"
                    },
                    "print_solve_stats": 1, 
                    "monitor_residual": 1,
                    "solver": "PCG", 
                    "convergence": "ABSOLUTE",
                    "tolerance": 1e-5,
                    "max_iters": 200
                    }
        })
        
        
        
        resources = pyamgx.Resources().create_simple(cfg)
        
        # Create Matrices & Vectors
        vec = pyamgx.Vector().create(resources, mode='dDDI')
        mat = pyamgx.Matrix().create(resources, mode='dDDI')
        x = pyamgx.Vector().create(resources, mode='dDDI')
        
        # Construct Linear System
        Smat1 = scipy.sparse.csr_matrix((self._ss * (1.0 / np.square(kl)),
                                            (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        
        Smat_O1 = scipy.sparse.identity(self._n2d, dtype=np.float64, format='csr') + 0.5 * (Smat1 ** 1) #n)      
        
        sol = np.zeros(self._n2d, dtype=np.float64)
        
        #ttw = ttu - Smat @ ttu
                
        # Create Solver
        solver = pyamgx.Solver().create(resources, cfg, mode='dDDI')
        
        # Upload System
        mat.upload_CSR(Smat_O1)
        vec.upload(ttu._value.astype(np.float64))
        x.upload(sol)

        # Setup & Solve O1 System (for Initial Guess)
        solver.setup(mat)
        solver.solve(vec, x)
        
        
        
        
        ## Upload O2 System
        Smat_O2 = scipy.sparse.identity(self._n2d, dtype=np.float64, format='csr') + 0.5 * (Smat1 ** 2) #n)
        mat.upload_CSR(Smat_O2)
        # Setup & Solve O2 System
        solver.setup(mat)
        solver.solve(vec, x)
        
        
        # Download Solution
        x.download(sol)
        
        # Clean Up
        x.destroy()
        solver.destroy()
        mat.destroy()
        vec.destroy()
        resources.destroy()
        cfg.destroy()
        
        pyamgx.finalize()
        
        
        #sol += ttu
        
        return sol

