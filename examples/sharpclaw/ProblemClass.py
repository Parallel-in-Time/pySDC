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
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
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
        assert 'nu' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(sharpclaw,self).__init__(self.nvars,dtype_u,dtype_f)

        # compute dx and get discretization matrix A
        self.dx = 1./(self.nvars + 1.)
        self.A = self.__get_A(self.nvars,self.nu,self.dx)
            
        # At the moment, there is no interaction of these lines with the rest of the code
        riemann_solver         = riemann.advection_1D # NOTE: This uses the FORTRAN kernels of clawpack
        solver                 = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.weno_order      = 5
        solver.time_integrator = 'Euler' # Remove later
        solver.kernel_language = 'Fortran'
        solver.bc_lower[0]     = pyclaw.BC.periodic
        solver.bc_upper[0]     = pyclaw.BC.periodic
        
        x      = pyclaw.Dimension(0.0,1.0,self.nvars,name='x')
        domain = pyclaw.Domain(x)
        state  = pyclaw.State(domain,solver.num_eqn)
        state.problem_data['u'] = 1.0
            
        # Initial data
        xc = state.grid.x.centers
        beta = 100; gamma=0; x0 = 0.75
        state.q[0,:] = np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))

        claw = pyclaw.Controller()
        claw.keep_copy = True
        claw.solution = pyclaw.Solution(state,domain)
        claw.solver = solver
        claw.outdir = './_output'
        claw.tfinal = 1.0
   
        my_state = claw.solution.states[0]
        solver.setup(claw.solution)
        solver.dt = 0.001
        solver.cfl_max = 1.0
        print solver.is_valid()
        print solver.cfl_max
        deltaq   = solver.dq(my_state)
        print np.size(deltaq)
        # Note: A forward Euler step would now read state.q += deltaq
        # ..cf line 262ff in pyclaw/sharpclaw/solver.py
        claw.run()

    def __get_A(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil,[-1,0,1],shape=(N,N))
        A *= nu / (dx**2)
        return A.tocsc()


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
        me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A,rhs.values)
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

        xvalues = np.array([(i+1)*self.dx for i in range(self.nvars)])
        fexpl = mesh(self.nvars)
        fexpl.values = -np.sin(np.pi*xvalues)*(np.sin(t)-self.nu*np.pi**2*np.cos(t))
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

        fimpl = mesh(self.nvars)
        fimpl.values = self.A.dot(u.values)
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
        me.values = np.sin(np.pi*xvalues)*np.cos(t)
        return me
