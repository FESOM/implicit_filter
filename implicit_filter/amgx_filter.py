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
        
        cfg = pyamgx.Config()
        #cfg.create_from_file('/home/b/b382615/opt/AMGX_test_config.json')
        cfg.create_from_dict(self._config)
        # cfg = pyamgx.Config().create_from_dict({
        #     "config_version": 2,
        #             "determinism_flag": 1,
        #             "exception_handling" : 1,
        #             "solver": {
        #                 "monitor_residual": 1,
        #                 "solver": "BICGSTAB",
        #                 "convergence": "RELATIVE_INI_CORE",
        #                 "preconditioner": {
        #                     "solver": "NOSOLVER"
        #             }
        #         }
        #     })
        
        resources = pyamgx.Resources()
        resources.create_simple(cfg)
        
        vec = pyamgx.Vector()
        vec.create(resources, mode='dFFI')
        vec.upload(ttu._value)
        # NOTE: Must be float32 !
        # ptr = ttu._value.__array_interface__['data'][0]
        # buff = ttu._value.shape[0]
        # vec.upload_raw(ptr, buff)
        
        mat = pyamgx.Matrix()
        mat.create(resources, mode='dFFI')
        
        Smat1 = scipy.sparse.csr_matrix((self._ss * (1.0 / np.square(kl)),
                                            (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        
        Smat = scipy.sparse.identity(self._n2d, dtype=np.float32) + 0.5 * (Smat1 ** n)        
        mat.upload_CSR(Smat)
        #mat.upload(Smat.indptr, Smat.indices, Smat.data, shape=[self._n2d, self._n2d])
        
        
        solver = pyamgx.Solver()
        solver.create(resources, cfg, mode='dFFI')
        
        solver.setup(mat)
        
        # Solve
        sol = np.zeros(self._n2d, dtype=np.float32)
        x = pyamgx.Vector().create(resources, mode='dFFI')
        x.upload(sol)
        solver.solve(vec, x)
        
        x.download(sol)
        
        
        x.destroy()
        solver.destroy()
        mat.destroy()
        vec.destroy()
        resources.destroy()
        cfg.destroy()
        
        pyamgx.finalize()
        
        return sol

