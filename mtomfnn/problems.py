# -*- coding: utf-8 -*-

"""Topology optimization problem to solve."""
import abc

import numpy
import scipy.sparse
import scipy.sparse.linalg
import cvxopt
import cvxopt.cholmod

from .boundary_conditions import BoundaryConditions
from .micro_structures import MicroStruct
from .utils import deleterowcol

class Problem(abc.ABC):
    def __init__(self, bc: BoundaryConditions, micro: MicroStruct): 
        # Problem size
        self.nelx = bc.nelx
        self.nely = bc.nely
        self.nel = self.nelx * self.nely

        # Count degrees of fredom
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # microstructre 
        self.micro = micro

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # RHS and Solution vectors
        self.f = bc.forces
        self.u = numpy.zeros(self.f.shape)

        # Per element objective
        self.obje = numpy.zeros(self.nely * self.nelx)
        self.dobj = numpy.zeros(self.nely * self.nelx)
        self.dv = numpy.ones(self.nely * self.nelx)
        # microStructure information 
        self.CE = None
        self.CE_Diff = None
        

    def __str__(self) -> str:
        """Create a string representation of the problem."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Create a formated representation of the problem."""
        return str(self)

    def __repr__(self) -> str:
        """Create a representation of the problem."""
        return "{}(bc={!r})".format(
            self.__class__.__name__, self.bc)


    @abc.abstractmethod
    def compute_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        
        pass


class ElasticityProblem(Problem):
    def __init__(self, bc: BoundaryConditions, micro: MicroStruct):
       
        super().__init__(bc, micro)
        # Max and min stiffness
        self.Emin = 1e-9
        self.Emax = 1.0

        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.3
        self.build_indices()

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # Number of loads
        self.nloads = self.f.shape[1]

    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        self.edofMat = numpy.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely # for the el-th element
                n1 = (self.nely + 1) * elx + ely # 左上角node
                n2 = (self.nely + 1) * (elx + 1) + ely # 右上角node
                self.edofMat[el, :] = numpy.array([
                    2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2,
                    2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((8, 1), dtype=int)).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 8), dtype=int)).flatten()

    def _updateConstitutiveInformation(self, xPhys: numpy.ndarray) -> None:
        self.CE, self.CE_Diff = self.micro.getCE_and_CEDiff(xPhys)
    

    def _gaussian_integral(self, CE):
        res = numpy.zeros((8, 8))
        L = numpy.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 1, 0]])
        GN_x = numpy.array([-1 / numpy.sqrt(3), 1 / numpy.sqrt(3)])
        GN_y = GN_x.copy()
        GaussWeigh=[1, 1]
        dN = numpy.zeros((4, 8))
        for i in range(len(GN_x)):
            for j in range(len(GN_y)):
                x = GN_x[i]
                y = GN_y[j]
                dNx = 1/4 * numpy.array([-(1-y), (1-y), (1+y), -(1+y)])
                dNy = 1/4 * numpy.array([-(1-x), -(1+x), (1+x), (1-x)])
                dN[0,0:8:2] = dNx
                dN[1,0:8:2] = dNy
                dN[2,1:8:2] = dNx
                dN[3,1:8:2] = dNy
              
                Be = L @ dN
                res = res + GaussWeigh[i] * GaussWeigh[j] *(Be.T @ CE @ Be) 
        return res
    

    def _get_KE(self, CE_e):
        C11, C12, C33 = CE_e
        CE = numpy.array([[C11, C12, 0],
                       [C12, C11, 0],
                       [0, 0, C33]])
        
        return self._gaussian_integral(CE)

    def _get_DKE(self, CE_Diff_e):
        C11, C12, C33 = CE_Diff_e
        CE_Diff = numpy.array([[C11, C12, 0],
                       [C12, C11, 0],
                       [0, 0, C33]])
        return self._gaussian_integral(CE_Diff)
        
        

    def build_K(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
        self._updateConstitutiveInformation(xPhys)
        sK = []
        for CE_e in self.CE:
            sK.append(self._get_KE(CE_e).flatten())
        sK = numpy.array(sK).flatten()
        K = scipy.sparse.coo_matrix(
            (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            K = deleterowcol(K.tocsc(), self.fixed, self.fixed).tocoo()
        return K

        
      
    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        K = self.build_K(xPhys)
        K = cvxopt.spmatrix(
            K.data, K.row, K.col)
       
        # Solve system
        F = cvxopt.matrix(self.f[self.free, :])
        cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
        new_u = self.u.copy()
        new_u[self.free, :] = numpy.array(F)[:, :]
        return new_u

    def update_displacements(self, xPhys: numpy.ndarray) -> None:
        self.u[:, :] = self.compute_displacements(xPhys)
       


class ComplianceProblem(ElasticityProblem):

    def compute_objective(self, xPhys: numpy.ndarray):
       # Setup and solve FE problem
        self.update_displacements(xPhys)

        obj = 0.0
        self.dobj[:] = 0.0
        for i in range(self.nloads):      
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            for j in range(numpy.size(xPhys)):
                KE = self._get_KE(self.CE[j])
                DKE = self._get_DKE(self.CE_Diff[j])
                if ui[j] @ DKE @ ui[j] < 0:
                    print('DKE Error')
                self.obje[j] = ui[j] @ KE @ ui[j] 
                self.dobj[j] = numpy.min((-1 * ui[j] @ DKE @ ui[j],0))
                obj +=   self.obje[j]          
        self.dobj /= float(self.nloads)
        return obj / float(self.nloads)

