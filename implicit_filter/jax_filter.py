import concurrent
from typing import Tuple, Union, List
import numpy as np
import jax.numpy as jnp
from jax import vmap
from ._auxiliary import neighboring_triangles, neighbouring_nodes, areas
from ._jax_function import make_smooth, make_smat, make_smat_full, transform_vector_to_nodes, transform_to_nodes, transform_vector_to_cells, transform_to_cells
from ._utils import VeryStupidIdeaError, SolverNotConvergedError
from implicit_filter.filter import Filter
from scipy.sparse import csc_matrix, identity, spdiags
from scipy.sparse.linalg import cg
import xarray as xr


class JaxFilter(Filter):
    """
    A class for filtering data using JAX for accelerating implicit filtering techniques.
    Extends the base Filter class.

    Attributes:
    -----------
    _elem_area : Optional[jnp.ndarray]
        Area of each element in the mesh.
    _area : Optional[jnp.ndarray]
        Area of each node's neighborhood in the mesh.
    _ne_pos : Optional[jnp.ndarray]
        Connectivity matrix representing neighboring elements for each node.
    _ne_num : Optional[jnp.ndarray]
        Number of neighboring elements for each node.
    _dx : Optional[jnp.ndarray]
        X-component of the derivative of P1 basis functions.
    _dy : Optional[jnp.ndarray]
        Y-component of the derivative of P1 basis functions.
    _ss : Optional[jnp.ndarray]
        Non-zero entries of the sparse matrix.
    _ii : Optional[jnp.ndarray]
        Row indices of non-zero entries.
    _jj : Optional[jnp.ndarray]
        Column indices of non-zero entries.
    _n2d : int
        Total number of nodes in the mesh.
    _full : bool
        Flag indicating whether to use the full matrix.

    Methods:
    --------
    compute_velocity(n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute filtered velocity components (u, v) using implicit filtering.
    compute(n: int, k: float, data: np.ndarray) -> np.ndarray:
        Compute filtered data using implicit filtering.
    prepare(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
            carthesian: bool, cyclic_length: float, full: bool = False):
        Prepare the filter for a specific mesh.

    """

    def __transform_atribute(self, atr: str, lmbd, fill=None):
        """
        If atribute atr exists then transform it using given Callable lmbd, otherwise it set with fill value
        """
        if hasattr(self, atr):
            setattr(self, atr, lmbd(getattr(self, atr)))
        else:
            setattr(self, atr, fill)

    def __init__(self, *initial_data, **kwargs):
        """
        Initialize the JaxFilter instance.

        Parameters:
        -----------
        *initial_data : positional arguments
            Positional arguments passed to the base class constructor.
        **kwargs : keyword arguments
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(initial_data, kwargs)
        # Transform to JAX array
        jx = lambda ar: jnp.array(ar)
        bl = lambda ar: bool(ar)
        it = lambda ar: int(ar)

        self.__transform_atribute("_elem_area", jx, None)
        self.__transform_atribute("_area", jx, None)
        self.__transform_atribute("_ne_pos", jx, None)
        self.__transform_atribute("_en_pos", jx, None)
        self.__transform_atribute("_ne_num", jx, None)
        self.__transform_atribute("_dx", jx, None)
        self.__transform_atribute("_dy", jx, None)

        self.__transform_atribute("_ss", jx, None)
        self.__transform_atribute("_ii", jx, None)
        self.__transform_atribute("_jj", jx, None)

        self.__transform_atribute("_n2d", it, 0)
        self.__transform_atribute("_e2d", it, 0)
        self.__transform_atribute("_full", bl, False)

    def _compute(self, n, kl, ttu, tol=1e-5, maxiter=150000) -> np.ndarray:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)

        # b = Smat.diagonal()
        # pre = csc_matrix((b, (np.arange(self._n2d), np.arange(self._n2d))), shape=(self._n2d, self._n2d))
        ttw = ttu - Smat @ ttu  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged without metric terms",
                                          [f"output code with code: {code}"])

        tts += ttu
        return np.array(tts)

    def _many_compute(self, n, kl, data, tol=1e-5, maxiter=150000) -> List[np.ndarray]:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
        Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)
        output: List[np.ndarray] = list()
        for ttu in data:
            ttw = ttu - Smat @ ttu  # Work with perturbations
            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)

            if code != 0:
                raise SolverNotConvergedError("Solver has not converged without metric terms",
                                              [f"output code with code: {code}"])

            tts += ttu
            output.append(tts)

        return output
    
    def _spectra_compute(self, n, kl, ttu, tol=1e-5, maxiter=150000) -> List[np.ndarray]:
        output: List[np.ndarray] = list()
        for kl_i in kl:
            Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl_i)), (self._ii, self._jj)), shape=(self._n2d, self._n2d))
            Smat = identity(self._n2d) + 0.5 * (Smat1 ** n)
            ttw = ttu - Smat @ ttu  # Work with perturbations
            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
            
            if code != 0:
                raise SolverNotConvergedError("Solver has not converged without metric terms",
                                              [f"output code with code: {code}"])

            tts += ttu
            output.append(tts)

        return output

    def _compute_full(self, n, kl, ttuv, tol=1e-5, maxiter=150000) -> np.ndarray:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = identity(2 * self._n2d) + 0.5 * (Smat1 ** n)

        ttw = ttuv - Smat @ ttuv  # Work with perturbations

        tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
        if code != 0:
            raise SolverNotConvergedError("Solver has not converged with metric terms",
                                          [f"output code with code: {code}"])

        tts += ttuv
        return np.array(tts)

    def _many_compute_full(self, n, kl, ux, vy, tol=1e-5, maxiter=150000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl)), (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d))
        Smat = identity(2 * self._n2d) + 0.5 * (Smat1 ** n)

        oux: List[np.ndarray] = list()
        ovy: List[np.ndarray] = list()

        for i in range(len(ux)):
            ttuv = jnp.concatenate((ux[i], vy[i]))
            ttw = ttuv - Smat @ ttuv  # Work with perturbations

            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
            if code != 0:
                raise SolverNotConvergedError("Solver has not converged with metric terms",
                                              [f"output code with code: {code}"])

            tts += ttuv

            oux.append(tts[0:self._n2d])
            ovy.append(tts[self._n2d:2 * self._n2d])

        return oux, ovy
    
    def _spectra_compute_full(self, n, kl, ux, vy, tol=1e-5, maxiter=150000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        oux: List[np.ndarray] = list()
        ovy: List[np.ndarray] = list()
        
        ttuv = jnp.concatenate((ux, vy))

        for kl_i in kl:
            Smat1 = csc_matrix((self._ss * (1.0 / jnp.square(kl_i)), (self._ii, self._jj)), shape=(2 * self._n2d, 2 * self._n2d))
            Smat = identity(2 * self._n2d) + 0.5 * (Smat1 ** n)
            
            ttw = ttuv - Smat @ ttuv  # Work with perturbations

            tts, code = cg(Smat, ttw, tol=tol, maxiter=maxiter)
            if code != 0:
                raise SolverNotConvergedError("Solver has not converged with metric terms",
                                              [f"output code with code: {code}"])

            tts += ttuv

            oux.append(tts[0:self._n2d])
            ovy.append(tts[self._n2d:2 * self._n2d])

        return oux, ovy

    def compute_velocity(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn, vyn = transform_vector_to_nodes(jnp.array(ux), jnp.array(vy), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)
        if self._full:
            ttuv = self._compute_full(n, k, jnp.concatenate((uxn, vyn)))
            return ttuv[0:self._n2d], ttuv[self._n2d:2*self._n2d]
        else:
            ttu = self._compute(n, k, uxn)
            ttv = self._compute(n, k, vyn)
            return ttu, ttv
    
    def compute_on_cells(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        data_n = transform_to_nodes(jnp.array(data), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)

        data_n_filtered = self._compute(n, k, data_n)
        data_cell_filtered = transform_to_cells(jnp.array(data_n_filtered), self._en_pos, self._e2d, self._elem_area)
        return data_cell_filtered
    
    def compute_vector_on_cells(self, n: int, k: float, ux: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn, vyn = transform_vector_to_nodes(jnp.array(ux), jnp.array(vy), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)
        if self._full:
            ttuv = self._compute_full(n, k, jnp.concatenate((uxn, vyn)))
            ttu = ttuv[0:self._n2d]
            ttv = ttuv[self._n2d:2*self._n2d]
        else:
            ttu = self._compute(n, k, uxn)
            ttv = self._compute(n, k, vyn)
            
        ttu_centre, ttv_centre = transform_vector_to_cells(jnp.array(ttu), jnp.array(ttv), self._en_pos, self._e2d, self._elem_area)

        return ttu_centre, ttv_centre

    def compute(self, n: int, k: float, data: np.ndarray) -> np.ndarray:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        return np.array(self._compute_full(n, k, data) if self._full else self._compute(n, k, data))

    def many_compute(self, n: int, k: float, data: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        if type(data) is np.ndarray:
            if len(data.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            return self._many_compute(n, k, data.T)
        elif type(data) is list:
            return self._many_compute(n, k, data)
        else:
            raise ValueError("Input data is of incorrect type")


    def many_compute_velocity(self, n: int, k: float, ux: Union[np.ndarray, List[np.ndarray]],
                              vy: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn = []
        vyn = []
        futures = []

        if type(ux) is np.ndarray:
            if len(ux.shape) != 2 and len(vy.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux.T)):
                    futures.append(executor.submit(transform_vector_to_nodes, jnp.array(ux[:, i]), jnp.array(vy[:, i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)

        elif type(ux) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux)):
                    futures.append(executor.submit(transform_vector_to_nodes, jnp.array(ux[i]), jnp.array(vy[i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")

        for f in futures:
            tmp_u, tmp_v = f.result()
            uxn.append(tmp_u)
            vyn.append(tmp_v)

        if self._full:
            ttu, ttv = self._many_compute_full(n, k, uxn, vyn)
        else:
            ttu = self._many_compute(n, k, uxn)
            ttv = self._many_compute(n, k, vyn)

        return ttu, ttv
    
    
    def many_compute_on_cells(self, n: int, k: float, data: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        data_n = []
        futures = []
        ttu_centre = []
        
        # Put onto nodes...
        if type(data) is np.ndarray:
            if len(data.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(data)):
                    futures.append(executor.submit(transform_to_nodes, jnp.array(data[i, :]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        
        elif type(data) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(data)):
                    futures.append(executor.submit(transform_to_nodes, jnp.array(data[i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")
        
        for f in futures:
            tmp_data = f.result()
            data_n.append(tmp_data)
        

        ttu = self._many_compute(n, k, data_n)
        
        futures = []
        # Put back onto cell centres...
        if type(ttu) is np.ndarray:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ttu.T)):
                    futures.append(executor.submit(transform_to_cells, jnp.array(ttu[:, i]),
                                                   self._en_pos, self._e2d, self._elem_area))
                executor.shutdown(wait=True)
        
        elif type(ttu) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ttu)):
                    futures.append(executor.submit(transform_to_cells, jnp.array(ttu[i]),
                                                   self._en_pos, self._e2d, self._elem_area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")
        
        for f in futures:
            tmp_data_c = f.result()
            ttu_centre.append(tmp_data_c)
        
        return ttu_centre
    
    def spectra_compute_on_cells(self, n: int, k: np.ndarray, data: np.ndarray) -> List[np.ndarray]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        data_n = []
        futures = []
        ttu_centre = []
        
        # Put onto nodes...
        data_n = transform_to_nodes(jnp.array(data), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)

        
        ttu = self._spectra_compute(n, k, data_n)
        
        # Put back onto cell centres...
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(len(ttu)):
                futures.append(executor.submit(transform_to_cells, jnp.array(ttu[i]),
                                                self._en_pos, self._e2d, self._elem_area))
            executor.shutdown(wait=True)
        
        for f in futures:
            tmp_data_c = f.result()
            ttu_centre.append(tmp_data_c)
        
        return ttu_centre
    
    
    def many_compute_vector_on_cells(self, n: int, k: float, ux: Union[np.ndarray, List[np.ndarray]], vy: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        elif n > 2:
            raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn = []
        vyn = []
        futures = []
        ttu_centre = []
        ttv_centre = []
        
        # Put onto nodes...
        if type(ux) is np.ndarray:
            if len(ux.shape) != 2:
                raise ValueError("Input NumPy array must be 2D")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux)):
                    futures.append(executor.submit(transform_vector_to_nodes, jnp.array(ux[i, :]), jnp.array(vy[i, :]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        
        elif type(ux) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ux)):
                    futures.append(executor.submit(transform_vector_to_nodes, jnp.array(ux[i]), jnp.array(vy[i]),
                                                   self._ne_pos, self._ne_num, self._n2d, self._elem_area, self._area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")
        
        for f in futures:
            tmp_u, tmp_v = f.result()
            uxn.append(tmp_u)
            vyn.append(tmp_v)
        
        if self._full:
            ttu, ttv = self._many_compute_full(n, k, uxn, vyn)
        else:
            ttu = self._many_compute(n, k, uxn)
            ttv = self._many_compute(n, k, vyn)
        
        futures = []
        # Put back onto cell centres...
        if type(ttu) is np.ndarray:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ttu.T)):
                    futures.append(executor.submit(transform_vector_to_cells, jnp.array(ttu[:, i]), jnp.array(ttv[:, i]),
                                                   self._en_pos, self._e2d, self._elem_area))
                executor.shutdown(wait=True)
        
        elif type(ttu) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(ttu)):
                    futures.append(executor.submit(transform_vector_to_cells, jnp.array(ttu[i]), jnp.array(ttv[i]),
                                                   self._en_pos, self._e2d, self._elem_area))
                executor.shutdown(wait=True)
        else:
            raise ValueError("Input data is of incorrect type")
        
        for f in futures:
            tmp_u_c, tmp_v_c = f.result()
            ttu_centre.append(tmp_u_c)
            ttv_centre.append(tmp_v_c)
        
        return ttu_centre, ttv_centre
    
    
    def spectra_compute_vector_on_cells(self, n: int, k: np.ndarray, ux: np.ndarray, vy: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if n < 1:
            raise ValueError("Filter order must be positive")
        # elif n > 2:
        #     raise VeryStupidIdeaError("Filter order too large", ["It really shouldn't be larger than 2"])

        uxn = []
        vyn = []
        futures = []
        ttu_centre = []
        ttv_centre = []
        
        # Put onto nodes...
        uxn, vyn = transform_vector_to_nodes(jnp.array(ux), jnp.array(vy), self._ne_pos, self._ne_num, self._n2d,
                                               self._elem_area, self._area)
        
        
        if self._full:
            ttu, ttv = self._spectra_compute_full(n, k, uxn, vyn)
        else:
            ttu = self._spectra_compute(n, k, uxn)
            ttv = self._spectra_compute(n, k, vyn)
        
        # Put back onto cell centres...
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(len(ttu)):
                futures.append(executor.submit(transform_vector_to_cells, jnp.array(ttu[i]), jnp.array(ttv[i]),
                                                self._en_pos, self._e2d, self._elem_area))
            executor.shutdown(wait=True)
        
        for f in futures:
            tmp_u_c, tmp_v_c = f.result()
            ttu_centre.append(tmp_u_c)
            ttv_centre.append(tmp_v_c)
        
        return ttu_centre, ttv_centre
    
    

    def prepare(self, n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, meshtype: str,
                carthesian: bool, cyclic_length: float, resolution: float, full: bool = False, mask: np.ndarray = None, L_Ro: np.ndarray = None):
        
        # NOTE: xcoord & ycoord are in degrees, but cyclic_length is in radians
        
        if mask is None:
            mask = np.ones(e2d)
        
        if L_Ro is None:
            L_Ro = np.ones(e2d)

        ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
        nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
        area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, carthesian,
                                        cyclic_length, mask)

        self._elem_area = jnp.array(elem_area)
        self._dx = jnp.array(dx)
        self._dy = jnp.array(dy)
        jMt = jnp.array(Mt)
        jnn_num = jnp.array(nn_num)
        jnn_pos = jnp.array(nn_pos)
        jtri = jnp.array(tri)
        self._en_pos = jnp.array(tri.T)  # element positions in terms of nodes
        self._ne_num = jnp.array(ne_num)
        self._ne_pos = jnp.array(ne_pos)
        self._area = jnp.array(area)
        
        smooth, metric = make_smooth(jMt, self._elem_area, self._dx, self._dy, jnn_num, jnn_pos, jtri, n2d, e2d, full)

        smooth = vmap(lambda n: smooth[:, n] / self._area[n])(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] / self._area[n])(jnp.arange(0, n2d)).T

        ## Scale by L_Ro — Then filter units are [L_Ro]  
        # AFW
        L_Ro_n = transform_to_nodes(L_Ro, self._ne_pos, self._ne_num, n2d,
                                        self._elem_area, self._area)
        L_Ro_n = jnp.array(L_Ro_n)
        smooth = vmap(lambda n: smooth[:, n] * L_Ro_n[n]**2 / resolution**2)(jnp.arange(0, n2d)).T
        metric = vmap(lambda n: metric[:, n] * L_Ro_n[n]**2 / resolution**2)(jnp.arange(0, n2d)).T
        
        
        ## Set smooth to zero (also for BCs) (?)
        mask_n = transform_to_nodes(mask, self._ne_pos, self._ne_num, n2d, self._elem_area, self._area)
        mask_n = jnp.where(mask_n > 0.5, 1.0, 0.0)
        
        # smooth = vmap(lambda n: smooth[:, n] * mask_n[n])(jnp.arange(0, n2d)).T
        # metric = vmap(lambda n: metric[:, n] * mask_n[n])(jnp.arange(0, n2d)).T
        
        self._ss, self._ii, self._jj = make_smat_full(jnn_pos, jnn_num, smooth, metric, n2d, int(jnp.sum(jnn_num))) \
            if full else make_smat(jnn_pos, jnn_num, smooth, n2d, int(jnp.sum(jnn_num)))
        
        ## Set rows (and columns!) of smooth where (node) mask is 0 (land) to 0: This enforces a Neumann BC
        #   i.e. Set _ss = 0 where mask_n[_ii] = 0 && mask_n[_jj] = 0
        # AFW
        
        mask_n = mask_n.astype(bool)
        
        # Create a mask where both _ii and _jj are not 0
        if full:
            mask = (mask_n[self._ii%n2d] & mask_n[self._jj%n2d])
        else:
            mask = (mask_n[self._ii] & mask_n[self._jj])

        self._ss = self._ss[mask]
        self._ii = self._ii[mask]
        self._jj = self._jj[mask]        
        
        
        self._n2d = n2d
        self._e2d = e2d
        self._full = full



    def prepare_ICON_filter(self, grid2d: xr.DataArray, land_mask: xr.DataArray = None, full: bool = False, L_Ro: xr.DataArray = None, L_Ro_max: float = 50.0, resolution: float = 5.0):
        
        # Prepare the mesh data
        xcoord = grid2d['vlon'].values * 180.0/np.pi
        ycoord = grid2d['vlat'].values * 180.0/np.pi  # Location of nodes, in degrees
        tri = grid2d['vertex_of_cell'].values.T - 1
        tri = tri.astype(int)
        
        if land_mask is None:
            mask = np.ones(len(tri[:,1]))
        else:
            mask = xr.where(land_mask.values < 0.0, 1.0, 0.0)
            # NOTE: LSM is in grid2d['cell_sea_land_mask'] or in grid3d['lsm_c'].isel(depth=???)
        
        if L_Ro is not None:
            
            # small: factor 5 (50-250 km)
            # LRfactor = 5
            # LRmax = 250
            # large: factor 90 (900-4500 km)
            # LRfactor = 90
            # LRmax = 4500
            # medium: factor 12 (120-600 km)
            # LRfactor = 12
            # LRmax = 600
            
            L_Ro = L_Ro.values   # L_Ro needs to be in units of [km]            
            L_Ro = np.where(L_Ro > L_Ro_max, L_Ro_max, L_Ro)
            L_Ro = np.where(L_Ro < 2.5, 2.5, L_Ro)
        
        self.prepare(len(xcoord), len(tri[:,1]), tri, xcoord , ycoord,  meshtype='r', carthesian=False, cyclic_length=2.0*np.pi, resolution=resolution, full=full, mask=mask, L_Ro=L_Ro)


    def filter_ICON(self, n: int, filter_length: Union[float, np.ndarray], ux: xr.DataArray, vy: xr.DataArray=None, mask: float=None) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
        dims = ux.dims
        coords = ux.coords
        
        k = 1.0 / filter_length
        
        if vy is None:  # ux is scalar data...
        
            if 'time' in dims:
                # Cycle through each time step in parallel
                data = ux.values
                filtered_x = self.many_compute_on_cells(n, k, data) # Returns list of np.array...
                filtered_x = np.array(filtered_x)
            elif isinstance(k, np.ndarray):
                # Cycle through each wavenumber in parallel
                dims = ('k', dims[0])
                coords = coords.assign({'k': k})
                data = ux.values
                filtered_x = self.spectra_compute_on_cells(n, k[::-1], data) # Returns list of np.array...
                filtered_x = filtered_x[::-1]   #np.array(filtered_x)
            else:
                # Just the one timestep...
                data = ux.values
                filtered_x = self.compute_on_cells(n, k, data)
            
            da_filtered_x = xr.DataArray(filtered_x, coords=coords, dims=dims)
            
            # Fix DataArray
            da_filtered_x.attrs = ux.attrs
            if mask is not None:
                da_filtered_x = xr.where(ux == mask, mask, da_filtered_x)
            
            return da_filtered_x
        
        else:  # ux and uy are vector data, so apply metric terms
            
            if 'time' in dims:
                # Cycle through each time step in parallel
                uxd = ux.values
                vyd = vy.values
                filtered_x, filtered_y = self.many_compute_vector_on_cells(n, k, uxd, vyd) # Returns list of np.array...
                filtered_x = np.array(filtered_x)
                filtered_y = np.array(filtered_y)
            elif isinstance(k, np.ndarray):
                # Cycle through each wavenumber in parallel
                dims = ('k', dims[0])
                coords = coords.assign({'k': k})
                uxd = ux.values
                vyd = vy.values
                filtered_x, filtered_y = self.spectra_compute_vector_on_cells(n, k[::-1], uxd, vyd)  # Invert k because last (smallest) k is the slowest, and want to do this first on GPUs
                filtered_x = filtered_x[::-1]
                filtered_y = filtered_y[::-1]
            else:
                # Just the one timestep...
                uxd = ux.values
                vyd = vy.values
                filtered_x, filtered_y = self.compute_vector_on_cells(n, k, uxd, vyd)
                
            da_filtered_x = xr.DataArray(filtered_x, coords=coords, dims=dims)
            da_filtered_y = xr.DataArray(filtered_y, coords=coords, dims=dims)
            
            # Fix DataArray
            da_filtered_x.attrs = ux.attrs
            da_filtered_y.attrs = vy.attrs
            if mask is not None:
                da_filtered_x = xr.where(ux == mask, mask, da_filtered_x)
                da_filtered_y = xr.where(vy == mask, mask, da_filtered_y)
        
            return da_filtered_x, da_filtered_y
    



