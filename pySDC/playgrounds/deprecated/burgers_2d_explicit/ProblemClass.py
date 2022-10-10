import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA
from clawpack import pyclaw
from clawpack import riemann

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes import mesh, rhs_imex_mesh


class sharpclaw(ptype):
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
        for k, v in cparams.items():
            setattr(self, k, v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(sharpclaw, self).__init__(self.nvars, dtype_u, dtype_f)

        riemann_solver = riemann.burgers_1D  # NOTE: This uses the FORTRAN kernels of clawpack
        self.solver = pyclaw.SharpClawSolver1D(riemann_solver)
        self.solver.weno_order = 5
        self.solver.time_integrator = 'Euler'  # Remove later
        self.solver.kernel_language = 'Fortran'
        self.solver.bc_lower[0] = pyclaw.BC.periodic
        self.solver.bc_upper[0] = pyclaw.BC.periodic
        self.solver.cfl_max = 1.0
        assert self.solver.is_valid()

        x = pyclaw.Dimension(0.0, 1.0, self.nvars, name='x')
        self.domain = pyclaw.Domain(x)

        self.state = pyclaw.State(self.domain, self.solver.num_eqn)
        self.dx = self.state.grid.x.centers[1] - self.state.grid.x.centers[0]

        solution = pyclaw.Solution(self.state, self.domain)
        self.solver.setup(solution)

        self.A = self.__get_A(self.nvars, self.nu, self.dx)

    def __get_A(self, N, nu, dx):
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
        A = sp.diags(stencil, [-1, 0, 1], shape=(N, N))
        A *= nu / (dx**2)
        return A.tocsc()

    def solve_system(self, rhs, factor, u0, t):
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

        me = mesh(self.nvars)
        me.values = LA.spsolve(sp.eye(self.nvars) - factor * self.A, rhs.values)

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        # Copy values of u into pyClaw state object
        self.state.q[0, :] = u.values

        # Evaluate right hand side
        fexpl = mesh(self.nvars)
        fexpl.values = self.solver.dqdt(self.state)

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl = mesh(self.nvars, val=0.0)
        fimpl.values = self.A.dot(u.values)

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = rhs_imex_mesh(self.nvars)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        xc = self.state.grid.x.centers
        me = mesh(self.nvars)
        me.values = np.sin(2.0 * np.pi * (xc - t))
        return me
