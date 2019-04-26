import numpy as np
import scipy.sparse.linalg as LA
from build2DFDMatrix import get2DMesh
from buildWave2DMatrix import getWave2DMatrix, getWave2DUpwindMatrix
from unflatten import unflatten

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes import mesh, rhs_imex_mesh


class acoustic_2d_imex(ptype):
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
        assert 'c_s' in cparams
        assert 'u_adv' in cparams
        assert 'x_bounds' in cparams
        assert 'z_bounds' in cparams
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(acoustic_2d_imex,self).__init__(self.nvars,dtype_u,dtype_f)
                
        self.N     = [ self.nvars[1],  self.nvars[2] ]
        
        self.bc_hor = [ ['periodic', 'periodic'] , ['periodic', 'periodic'], ['periodic', 'periodic'] ]
        self.bc_ver = [ ['neumann', 'neumann'] ,  ['dirichlet', 'dirichlet'], ['neumann', 'neumann'] ]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.x_bounds, self.z_bounds, self.bc_hor[0], self.bc_ver[0])
       
        self.Id, self.M = getWave2DMatrix(self.N, self.h, self.bc_hor, self.bc_ver)
        self.D_upwind   = getWave2DUpwindMatrix( self.N, self.h[0] )
    
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
        sol, info =  LA.gmres( self.Id + factor*self.c_s*self.M, b, x0=u0.values.flatten(), tol=1e-13, restart=10, maxiter=20)
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
        fexpl = mesh(self.nvars)
        temp  = u.values.flatten()
        temp  = self.D_upwind.dot(temp)
        # NOTE: M_adv = -D_upwind, therefore add a minus here
        fexpl.values = unflatten(-self.u_adv*temp, 3, self.N[0], self.N[1])
              
        #fexpl.values = np.zeros((3, self.N[0], self.N[1]))
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
        fimpl = mesh(self.nvars,val=0.0)
        # NOTE: M = -A, therefore add a minus here
        fimpl.values = unflatten(-self.c_s*temp, 3, self.N[0], self.N[1])
        
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
        #me.values[2,:,:] = 0.5*np.exp(-0.5*( self.xx-self.c_s*t - self.u_adv*t )**2/0.2**2.0) + 0.5*np.exp(-0.5*( self.xx + self.c_s*t - self.u_adv*t)**2/0.2**2.0)
        me.values[2,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.15**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.15**2)
        return me
