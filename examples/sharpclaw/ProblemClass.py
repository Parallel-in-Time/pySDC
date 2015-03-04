import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

# Sharpclaw imports
from clawpack import pyclaw
from clawpack import riemann

class sharpclaw(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
      solver: A sharpclaw solver
      state: A ...
      domain: A ...
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
        assert 'dt' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(sharpclaw,self).__init__(self.nvars,dtype_u,dtype_f)

        # compute dx and get discretization matrix A
        self.dx = 1./(self.nvars + 1.)
        
        riemann_solver              = riemann.advection_1D # NOTE: This uses the FORTRAN kernels of clawpack
        self.solver                 = pyclaw.SharpClawSolver1D(riemann_solver)
        self.solver.weno_order      = 5
        self.solver.time_integrator = 'Euler' # Remove later
        self.solver.kernel_language = 'Fortran'
        self.solver.bc_lower[0]     = pyclaw.BC.periodic
        self.solver.bc_upper[0]     = pyclaw.BC.periodic
        self.solver.dt              = self.dt
        self.solver.cfl_max         = 1.0
        assert self.solver.is_valid()

        x           = pyclaw.Dimension(0.0,1.0,self.nvars,name='x')
        self.domain = pyclaw.Domain(x)

        self.state                   = pyclaw.State(self.domain, self.solver.num_eqn)
        self.state.problem_data['u'] = 1.0
  
        # Initial data
        xc = self.state.grid.x.centers
        beta = 100; gamma=0; x0 = 0.75
        
        self.state.q[0,:] = np.sin(np.pi*xc)
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

        me        = mesh(self.nvars)
        me.values = rhs.values
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

        # Copy values of u into pyClaw state object
        self.state.q[0,:] = u.values
        
        # Evaluate right hand side
        deltaq           = self.solver.dq(self.state)
        
        # Copy right hand side values back into pySDC solution structure
        fexpl        = mesh(self.nvars)
        fexpl.values = deltaq
        
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

        fimpl        = mesh(self.nvars)
        fimpl.values = 0.0*u.values
        
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

        me = mesh(self.nvars)
        xvalues = np.array([(i+1)*self.dx for i in range(self.nvars)])
        me.values = np.sin(np.pi*xvalues - t)
        return me
