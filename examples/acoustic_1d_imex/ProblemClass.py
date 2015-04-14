r"""
  One-dimensional IMEX acoustic-advection
  =========================
  
  Integrate the linear 1D acoustic-advection problem:
  
  .. math::
  u_t + U u_x + c p_x & = 0 \\
  p_t + U p_x + c u_x & = 0.
  
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

# Sharpclaw imports
from clawpack import pyclaw
from clawpack import riemann

from getFDMatrix import getFDMatrix

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
        
        riemann_solver              = riemann.advection_1D # NOTE: This uses the FORTRAN kernels of clawpack
        self.solver                 = pyclaw.SharpClawSolver1D(riemann_solver)
        self.solver.weno_order      = 5
        self.solver.time_integrator = 'Euler' # Remove later
        self.solver.kernel_language = 'Fortran'
        self.solver.bc_lower[0]     = pyclaw.BC.periodic
        self.solver.bc_upper[0]     = pyclaw.BC.periodic
        self.solver.cfl_max         = 1.0
        assert self.solver.is_valid()

        x = pyclaw.Dimension(0.0, 1.0, self.nvars[1], name='x')
        self.domain = pyclaw.Domain(x)

        self.state = pyclaw.State(self.domain, self.solver.num_eqn)
        self.mesh  = self.state.grid.x.centers
        self.dx    = self.mesh[1] - self.mesh[0]
        self.A     = -self.cs*getFDMatrix(self.nvars[1], self.order_adv, self.dx)

        self.state.problem_data['u'] = self.cadv
        
        solution = pyclaw.Solution(self.state, self.domain)
        self.solver.setup(solution)


    def solve_system(self,rhs,factor,u0):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        me.values[0,:] = LA.spsolve(sp.eye(self.nvars[1])-factor*self.A,rhs.values[1,:])
        me.values[1,:] = LA.spsolve(sp.eye(self.nvars[1])-factor*self.A,rhs.values[0,:])
        
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


        fexpl        = mesh(self.nvars)

        # Copy values of u into pyClaw state object
        self.state.q[0,:] = u.values[0,:]

        # Evaluate right hand side
        tmp = self.solver.dqdt(self.state)
        fexpl.values[0,:] = tmp.reshape(self.nvars[1:])

        # Copy values of u into pyClaw state object
        self.state.q[0,:] = u.values[1,:]

        # Evaluate right hand side
        tmp = self.solver.dqdt(self.state)
        fexpl.values[1,:] = tmp.reshape(self.nvars[1:])
        
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

        fimpl             = mesh(self.nvars,val=0)
        fimpl.values[0,:] = self.A.dot(u.values[1,:])
        fimpl.values[1,:] = self.A.dot(u.values[0,:])
        
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
        
        me             = mesh(self.nvars)
        me.values[0,:] = np.exp(-0.5*(self.mesh-0.5)**2/0.1**2)
        me.values[1,:] = 0.0*self.mesh

        return me
