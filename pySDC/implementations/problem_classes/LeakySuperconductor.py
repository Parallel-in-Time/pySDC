import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class LeakySuperconductor(ptype):
    """
    This is a toy problem to emulate a magnet that has been cooled to temperatures where superconductivity is possible.
    However, there is a leak! Some point in the domain is constantly heated and when this has heated up its environment
    sufficiently, there will be a runaway effect heating up the entire magnet.
    This effect has actually lead to huge magnets being destroyed at CERN in the past and hence warrants investigation.

    The model we use is a 1d heat equation with Neumann-zero boundary conditions, meaning this magnet is totally
    insulated from its environment except for the leak.
    We add a non-linear term that heats parts of the domain that exceed a certain temperature threshold as well as the
    leak itself.
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        defaults = {
            'Cv': 1000.0,
            'K': 1000.0,
            'u_thresh': 1e-2,
            'u_max': 2e-2,
            'Q_max': 1.0,
            'leak_range': (0.45, 0.55),
            'order': 2,
            'stencil_type': 'center',
            'bc': 'neumann-zero',
            'nvars': 2**7,
            'direct_solver': True,
            'liniter': 99,
            'lintol': 1e-8,
            'newton_tol': 1e-8,
            'newton_iter': 99,
        }

        for key in problem_params.keys():
            if key not in defaults.keys():
                raise ParameterError(
                    f'Don\'t know what to do with parameter {key} in this problem!. Available parameters: {defaults}'
                )
        params = {**defaults, **problem_params}

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(
            init=(params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=params,
        )

        # compute dx (equal in both dimensions) and get discretization matrix A
        if self.params.bc == 'periodic':
            self.dx = 1.0 / self.params.nvars
            xvalues = np.array([i * self.dx for i in range(self.params.nvars)])
        elif self.params.bc == 'dirichlet-zero':
            self.dx = 1.0 / (self.params.nvars + 1)
            xvalues = np.array([(i + 1) * self.dx for i in range(self.params.nvars)])
        elif self.params.bc == 'neumann-zero':
            self.dx = 1.0 / (self.params.nvars - 1)
            xvalues = np.array([(i + 1) * self.dx for i in range(self.params.nvars)])
        else:
            raise ProblemError(f'Boundary conditions {self.params.bc} not implemented.')

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.params.order,
            type=self.params.stencil_type,
            dx=self.dx,
            size=self.params.nvars,
            dim=1,
            bc=self.params.bc,
        )
        self.A *= self.params.K

        self.xv = xvalues
        self.Id = sp.eye(np.prod(self.params.nvars), format='csc')

        self.leak = np.logical_and(self.xv > self.params.leak_range[0], self.xv < self.params.leak_range[1])

    def eval_f_non_linear(self, u, t):
        """
        Get the non-linear part of f

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_u: the non-linear part of the RHS
        """
        u_thresh = self.params.u_thresh
        u_max = self.params.u_max
        Q_max = self.params.Q_max
        me = self.dtype_u(self.init)

        me[:] = (u - u_thresh) / (u_max - u_thresh) * Q_max
        me[u < u_thresh] = 0
        me[self.leak] = Q_max
        me[u >= u_max] = Q_max

        # boundary conditions
        me[0] = 0.0
        me[-1] = 0.0

        return me

    def eval_f(self, u, t):
        """
        Evaluate the full right hand side.

        Args:
            u (dtype_u): Current solution
            t (float): Current time

        Returns:
            dtype_f: The right hand side
        """
        f = self.dtype_f(self.init)
        f[:] = (self.A.dot(u.flatten()).reshape(self.params.nvars) + self.eval_f_non_linear(u, t)) / self.params.Cv
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver for (I-factor*f)(u) = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        def get_non_linear_Jacobian(u):
            """
            Evaluate the non-linear part of the Jacobian only

            Args:
                u (dtype_u): Current solution

            Returns:
                dtype_u: The derivative of the non-linear part of the solution w.r.t. to the solution.
            """
            u_thresh = self.params.u_thresh
            u_max = self.params.u_max
            Q_max = self.params.Q_max
            me = self.dtype_u(self.init)

            me[:] = Q_max / (u_max - u_thresh)
            me[u < u_thresh] = 0
            me[u > u_max] = 0
            me[self.leak] = 0

            # boundary conditions
            me[0] = 0.0
            me[-1] = 0.0

            me = self.dtype_u(self.init)
            return me

        u = self.dtype_u(u0)
        res = np.inf
        for _n in range(0, self.params.newton_iter):
            # assemble G such that G(u) = 0 at the solution of the step
            G = u - factor * self.eval_f(u, t) - rhs

            res = np.linalg.norm(G, np.inf)
            if res <= self.params.newton_tol or np.isnan(res):
                break

            # assemble Jacobian J of G
            J = self.Id - factor * (self.A + get_non_linear_Jacobian(u))

            # solve the linear system
            if self.params.direct_solver:
                delta = np.linalg.solve(J, G)
            else:
                delta = cg(J, G, x0=G * 0, tol=self.params.lintol, maxiter=self.params.liniter, atol=0,)[
                    0
                ].reshape(self.params.nvars)

            # update solution
            u = u - delta

        return u

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init, val=0)

        if t > 0:

            def eval_rhs(t, u):
                """
                Function to pass to `scipy.solve_ivp` to evaluate the full RHS

                Args:
                    t (float): Current time
                    u (numpy.1darray): Current solution

                Returns:
                    (numpy.1darray): RHS
                """
                return self.eval_f(u.reshape(self.init[0]), t).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        return me


class LeakySuperconductorIMEX(LeakySuperconductor):
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        super().__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(u.flatten()).reshape(self.params.nvars) / self.params.Cv
        f.expl[:] = self.eval_f_non_linear(u, t) / self.params.Cv

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*f_expl)(u) = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        if self.params.direct_solver:
            me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.params.nvars)
        else:
            me[:] = cg(
                self.Id - factor * self.A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=self.params.lintol,
                maxiter=self.params.liniter,
                atol=0,
            )[0].reshape(self.params.nvars)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            me[:] = super().u_exact(t, u_init, t_init)

        if t > 0:

            def eval_rhs(t, u):
                """
                Function to pass to `scipy.solve_ivp` to evaluate the full RHS

                Args:
                    t (float): Current time
                    u (numpy.1darray): Current solution

                Returns:
                    (numpy.1darray): RHS
                """
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        return me
