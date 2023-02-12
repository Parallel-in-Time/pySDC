import numpy as np
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class LorenzAttractor(ptype):
    """
    Simple script to run a Lorenz attractor problem.

    The Lorenz attractor is a system of three ordinary differential equations that exhibits some chaotic behaviour.
    It is well known for the "Butterfly Effect", because the solution looks like a butterfly (solve to Tend = 100 or
    so to see this with these initial conditions) and because of the chaotic nature.

    Since the problem is non-linear, we need to use a Newton solver.

    Problem and initial conditions do not originate from, 
    but were taken from doi.org/10.2140/camcos.2015.10.1
    
    TODO : add equations with parameters
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, sigma=10., rho=28., beta=8/3, 
                 newton_tol=1e-9, newton_maxiter=99):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
        """
        nvars = 3
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'sigma', 'rho', 'beta', 'newton_tol', 'newton_maxiter',
            localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        # abbreviations
        sigma = self.sigma
        rho = self.rho
        beta = self.beta

        f = self.dtype_f(self.init)

        f[0] = sigma * (u[1] - u[0])
        f[1] = rho * u[0] - u[1] - u[0] * u[2]
        f[2] = u[0] * u[1] - beta * u[2]
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system.
        Notice that I did not go through the trouble of inverting the Jacobian beforehand. If you have some time on your
        hands feel free to do that! In the current implementation it is inverted using `numpy.linalg.solve`, which is a
        bit more expensive than simple matrix-vector multiplication.

        Args:
            rhs (dtype_f): right-hand side for the linear system
            dt (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        # abbreviations
        sigma = self.sigma
        rho = self.rho
        beta = self.beta

        # start Newton iterations
        u = self.dtype_u(u0)
        res = np.inf
        for _n in range(0, self.newton_maxiter):
            # assemble G such that G(u) = 0 at the solution to the step
            G = np.array(
                [
                    u[0] - dt * sigma * (u[1] - u[0]) - rhs[0],
                    u[1] - dt * (rho * u[0] - u[1] - u[0] * u[2]) - rhs[1],
                    u[2] - dt * (u[0] * u[1] - beta * u[2]) - rhs[2],
                ]
            )

            # compute the residual and determine if we are done
            res = np.linalg.norm(G, np.inf)
            if res <= self.newton_tol or np.isnan(res):
                break

            # assemble Jacobian J of G
            J = np.array(
                [
                    [1.0 + dt * sigma, -dt * sigma, 0],
                    [-dt * (rho - u[2]), 1 + dt, dt * u[0]],
                    [-dt * u[1], -dt * u[0], 1.0 + dt * beta],
                ]
            )

            # solve the linear system for the Newton correction J delta = G
            delta = np.linalg.solve(J, G)

            # update solution
            u = u - delta

        return u

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to return initial conditions or to approximate exact solution using scipy.

        Args:
            t (float): current time
            u_init (pySDC.implementations.problem_classes.Lorenz.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            dtype_u: exact solution
        """
        me = self.dtype_u(self.init)

        if t > 0:

            def eval_rhs(t, u):
                """
                Evaluate the right hand side, but switch the arguments for scipy.

                Args:
                    t (float): Time
                    u (numpy.ndarray): Solution at time t

                Returns:
                    (numpy.ndarray): Right hand side evaluation
                """
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = 1.0
        return me
