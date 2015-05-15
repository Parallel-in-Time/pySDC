import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

from build2DFDMatrix import get2DMesh
from buildWave2DMatrix import getWave2DMatrix, getWaveBCHorizontal, getWaveBCVertical
from unflatten import unflatten

class acoustic_2d_implicit(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
      solver: Sharpclaw solver
      state:  Sharclaw state
      domain: Sharpclaw domain
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'nvars' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(acoustic_2d_implicit,self).__init__(self.nvars,dtype_u,dtype_f)
        
        self.N   = [ self.nvars[1],  self.nvars[2] ]
        self.x_b = [ -1.0, 1.0]
        self.z_b = [  0.0, 1.0]
  
        self.bc_hor = [ ['periodic', 'periodic'] , ['periodic', 'periodic'], ['periodic', 'periodic'] ]
        self.bc_ver = [ ['neumann', 'neumann'] ,  ['dirichlet', 'dirichlet'], ['neumann', 'neumann'] ]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.x_b, self.z_b, self.bc_hor[0], self.bc_ver[0])

        self.Id, self.M = getWave2DMatrix(self.N, self.h, self.bc_hor, self.bc_ver)
  
    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        b = rhs.values.flatten()
        # NOTE: A = -M, therefore solve Id + factor*M here
        sol, info =  LA.gmres( self.Id + factor*self.M, b, x0=u0.values.flatten(), tol=1e-13, restart=10, maxiter=20)
        me = mesh(self.nvars)
        me.values = unflatten(sol, 3, self.N[0], self.N[1])

        return me


    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """
        
        # Evaluate right hand side
        fexpl        = mesh(self.nvars)
        fexpl.values = 0.0*self.xx
                
        return fexpl

    def __eval_fimpl(self,u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        temp = u.values.flatten()
        temp = self.M.dot(temp)
        fimpl = mesh(self.nvars,val=0)
        # NOTE: M = -A, therefore add a minus here
        fimpl.values = unflatten(-temp, 3, self.N[0], self.N[1])
        
        return fimpl


    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = rhs_imex_mesh(self.nvars)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """
        
        me        = mesh(self.nvars)
        me.values[0,:,:] = 0.0*self.xx
        me.values[1,:,:] = 0.0*self.xx
        me.values[2,:,:] = 0.5*np.exp(-0.5*(self.xx-t)**2/0.1**2.0) + 0.5*np.exp(-0.5*(self.xx+t)**2/0.1**2.0)
        return me
