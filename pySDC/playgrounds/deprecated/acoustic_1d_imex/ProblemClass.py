r"""
  One-dimensional IMEX acoustic-advection
  =========================
  
  Integrate the linear 1D acoustic-advection problem:
  
  .. math::
  u_t + U u_x + c p_x & = 0 \\
  p_t + U p_x + c u_x & = 0.
  
"""

import numpy as np
import scipy.sparse.linalg as LA
from buildWave1DMatrix import getWave1DMatrix, getWave1DAdvectionMatrix

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh


def u_initial(x):
    return np.sin(x)
#    return np.exp(-0.5*(x-0.5)**2/0.1**2)

class acoustic_1d_imex(ptype):
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
        assert 'cs' in cparams
        assert 'cadv' in cparams
        assert 'order_adv' in cparams
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(acoustic_1d_imex,self).__init__(self.nvars,dtype_u,dtype_f)
        
        self.mesh   = np.linspace(0.0, 1.0, self.nvars[1], endpoint=False)
        self.dx     = self.mesh[1] - self.mesh[0]
        
        self.Dx     = -self.cadv*getWave1DAdvectionMatrix(self.nvars[1], self.dx, self.order_adv)
        self.Id, A  = getWave1DMatrix(self.nvars[1], self.dx, ['periodic','periodic'], ['periodic','periodic'])
        self.A      = -self.cs*A
                
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
        
        M  = self.Id - factor*self.A
        
        b = np.concatenate( (rhs.values[0,:], rhs.values[1,:]) )
        
        sol = LA.spsolve(M, b)

        me = mesh(self.nvars)
        me.values[0,:], me.values[1,:] = np.split(sol, 2)
        
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


        b = np.concatenate( (u.values[0,:], u.values[1,:]) )
        sol = self.Dx.dot(b)
        
        fexpl        = mesh(self.nvars)
        fexpl.values[0,:], fexpl.values[1,:] = np.split(sol, 2)

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

        b = np.concatenate( (u.values[0,:], u.values[1,:]) )
        sol = self.A.dot(b)
        
        fimpl             = mesh(self.nvars,val=0.0)
        fimpl.values[0,:], fimpl.values[1,:] = np.split(sol, 2)
        
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
        
        sigma_0 = 0.1
        k       = 7.0*2.0*np.pi
        x_0     = 0.75
        x_1     = 0.25
        
        me             = mesh(self.nvars)
        #me.values[0,:] = 0.5*u_initial(self.mesh - (self.cadv + self.cs)*t) + 0.5*u_initial(self.mesh - (self.cadv - self.cs)*t)
        #me.values[1,:] = 0.5*u_initial(self.mesh - (self.cadv + self.cs)*t) - 0.5*u_initial(self.mesh - (self.cadv - self.cs)*t)
        me.values[0,:] = np.exp(-np.square(self.mesh-x_0-self.cs*t)/(sigma_0*sigma_0)) + np.exp(-np.square(self.mesh-x_1-self.cs*t)/(sigma_0*sigma_0))*np.cos(k*(self.mesh-self.cs*t)/sigma_0)
        me.values[1,:] = me.values[0,:]
        return me


