import numpy as np
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


class LorenzAttractor(ptype):
    r"""
    Simple script to run a Lorenz attractor problem.

    The Lorenz attractor is a system of three ordinary differential equations (ODEs) that exhibits some chaotic behaviour.
    It is well known for the "Butterfly Effect", because the solution looks like a butterfly (solve to :math:`T_{end} = 100`
    or so to see this with these initial conditions) and because of the chaotic nature.

    Since the problem is non-linear, we need to use a Newton solver.

    Problem and initial conditions do not originate from, but were taken from doi.org/10.2140/camcos.2015.10.1

    The system of ODEs is given by

    .. math::
        \frac{d y_1(t)}{dt} = \sigma (y_2 (t) - y_1 (t)),

    .. math::
        \frac{d y_2(t)}{dt} = \rho y_1 (t) - y_2 (t) - y_1 (t) y_3 (t),

    .. math::
        \frac{d y_3(t)}{dt} = y_1 (t) y_2 (t) - \beta y_3 (t)

    with initial condition :math:`y(0) = (1, 1, 1)^{T}` for :math:`t \in [0, 1]`. The problem parameters for this problem are
    :math:`\sigma = 10`, :math:`\rho = 28` and :math:`\beta = 8/3`.

    Parameters
    ----------
    sigma : float, optional
        Parameter :math:`\sigma` of the problem.
    rho : float, optional
        Parameter :math:`\rho` of the problem.
    beta : float, optional
        Parameter :math:`\beta` of the problem.
    newton_tol : float, optional
        Tolerance for Newton for termination.
    newton_maxiter : int, optional
        Maximum number of iterations for Newton's method.

    Attributes
    ----------
    work_counter : dict
        Counts the iterations/nfev (here for Newton's method and the nfev for the right-hand side).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0, newton_tol=1e-9, newton_maxiter=99):
        """Initialization routine"""

        nvars = 3
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'sigma', 'rho', 'beta', 'newton_tol', 'newton_maxiter', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        # abbreviations
        sigma = self.sigma
        rho = self.rho
        beta = self.beta

        f = self.dtype_f(self.init)

        f[0] = sigma * (u[1] - u[0])
        f[1] = rho * u[0] - u[1] - u[0] * u[2]
        f[2] = u[0] * u[1] - beta * u[2]

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
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

            # assemble inverse of Jacobian J of G
            prefactor = 1.0 / (
                dt**3 * sigma * (u[0] ** 2 + u[0] * u[1] + beta * (-rho + u[2] + 1))
                + dt**2 * (beta * sigma + beta - rho * sigma + sigma + u[0] ** 2 + sigma * u[2])
                + dt * (beta + sigma + 1)
                + 1
            )
            J_inv = prefactor * np.array(
                [
                    [
                        beta * dt**2 + dt**2 * u[0] ** 2 + beta * dt + dt + 1,
                        beta * dt**2 * sigma + dt * sigma,
                        -(dt**2) * sigma * u[0],
                    ],
                    [
                        beta * dt**2 * rho + dt**2 * (-u[0]) * u[1] - beta * dt**2 * u[2] + dt * rho - dt * u[2],
                        beta * dt**2 * sigma + beta * dt + dt * sigma + 1,
                        dt**2 * sigma * (-u[0]) - dt * u[0],
                    ],
                    [
                        dt**2 * rho * u[0] - dt**2 * u[0] * u[2] + dt**2 * u[1] + dt * u[1],
                        dt**2 * sigma * u[0] + dt**2 * sigma * u[1] + dt * u[0],
                        -(dt**2) * rho * sigma + dt**2 * sigma + dt**2 * sigma * u[2] + dt * sigma + dt + 1,
                    ],
                ]
            )

            # solve the linear system for the Newton correction J delta = G
            delta = J_inv @ G

            # update solution
            u = u - delta
            self.work_counters['newton']()

        return u

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to return initial conditions or to approximate exact solution using scipy.

        Parameters
        ----------
        t : float
            Time at which the approximated exact solution is computed.
        u_init : pySDC.implementations.problem_classes.Lorenz.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            The approximated exact solution.
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
