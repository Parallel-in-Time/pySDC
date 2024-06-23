#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of systems test problem ODEs.


Reference :

Van der Houwen, P. J., & Sommeijer, B. P. (1991). Iterated Runge–Kutta methods
on parallel computers. SIAM journal on scientific and statistical computing,
12(5), 1000-1028.
"""
import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


class ProtheroRobinsonAutonomous(Problem):
    r"""
    Implement the Prothero-Robinson problem into autonomous form:

    .. math::
        \begin{eqnarray*}
            \frac{du}{dt} &=& -\frac{u^3-g(v)^3}{\epsilon} + \frac{dg}{dv}, &\quad u(0) = g(0),\\
            \frac{dv}{dt} &=& 1, &\quad v(0) = 0,
        \end{eqnarray*}

    with :math:`\epsilon` a stiffness parameter, that makes the problem more stiff
    the smaller it is (usual taken value is :math:`\epsilon=1e^{-3}`).
    Exact solution is given by :math:`u(t)=g(t),\;v(t)=t`, and this implementation uses
    :math:`g(t)=\cos(t)`.

    Implement also the non-linear form of this problem:

    .. math::
        \frac{du}{dt} = -\frac{u^3-g(v)^3}{\epsilon} + \frac{dg}{dv}, \quad u(0) = g(0).

    To use an other exact solution, one just have to derivate this class
    and overload the `g`, `dg` and `dg2` methods. For instance,
    to use :math:`g(t)=e^{-0.2t}`, define and use the following class:

    >>> class MyProtheroRobinson(ProtheroRobinsonAutonomous):
    >>>
    >>>     def g(self, t):
    >>>         return np.exp(-0.2 * t)
    >>>
    >>>     def dg(self, t):
    >>>         return (-0.2) * np.exp(-0.2 * t)
    >>>
    >>>     def dg2(self, t):
    >>>         return (-0.2) ** 2 * np.exp(-0.2 * t)

    Parameters
    ----------
    epsilon : float, optional
        Stiffness parameter. The default is 1e-3.
    nonLinear : bool, optional
        Wether or not to use the non-linear form of the problem. The default is False.
    newton_maxiter : int, optional
        Maximum number of Newton iteration in solve_system. The default is 200.
    newton_tol : float, optional
        Residuum tolerance for Newton iteration in solve_system. The default is 5e-11.
    stop_at_nan : bool, optional
        Wheter to stop or not solve_system when getting NAN. The default is True.

    Reference
    ---------
    A. Prothero and A. Robinson, On the stability and accuracy of one-step methods for solving
    stiff systems of ordinary differential equations, Mathematics of Computation, 28 (1974),
    pp. 145–162.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, epsilon=1e-3, nonLinear=False, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 2
        super().__init__((nvars, None, np.dtype('float64')))

        self.f = self.f_NONLIN if nonLinear else self.f_LIN
        self.dgInv = self.dgInv_NONLIN if nonLinear else self.dgInv_LIN
        self._makeAttributeAndRegister(
            'epsilon', 'nonLinear', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    # -------------------------------------------------------------------------
    # g function (analytical solution), and its first and second derivative
    # -------------------------------------------------------------------------
    def g(self, t):
        return np.cos(t)

    def dg(self, t):
        return -np.sin(t)

    def dg2(self, t):
        return -np.cos(t)

    # -------------------------------------------------------------------------
    # f(u,t) and Jacobian functions
    # -------------------------------------------------------------------------
    def f(self, u, t):
        raise NotImplementedError()

    def f_LIN(self, u, t):
        return -self.epsilon ** (-1) * (u - self.g(t)) + self.dg(t)

    def f_NONLIN(self, u, t):
        return -self.epsilon ** (-1) * (u**3 - self.g(t) ** 3) + self.dg(t)

    def dgInv(self, u, t):
        raise NotImplementedError()

    def dgInv_LIN(self, u, t, dt):
        e = self.epsilon
        g1, g2 = self.dg(t), self.dg2(t)
        return np.array([[1 / (dt / e + 1), (dt * g2 + dt * g1 / e) / (dt / e + 1)], [0, 1]])

    def dgInv_NONLIN(self, u, t, dt):
        e = self.epsilon
        g, g1, g2 = self.g(t), self.dg(t), self.dg2(t)
        return np.array(
            [[1 / (3 * dt * u**2 / e + 1), (dt * g2 + 3 * dt * g**2 * g1 / e) / (3 * dt * u**2 / e + 1)], [0, 1]]
        )

    # -------------------------------------------------------------------------
    # pySDC required methods
    # -------------------------------------------------------------------------
    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to return initial conditions or exact solutions.

        Parameters
        ----------
        t : float
            Time at which the exact solution is computed.
        u_init : dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        u : dtype_u
            The exact solution.
        """
        u = self.dtype_u(self.init)
        u[0] = self.g(t)
        u[1] = t
        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (one component).
        """

        f = self.dtype_f(self.init)
        u, t = u
        f[0] = self.f(u, t)
        f[1] = 1
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Time of the updated solution (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)

        # start newton iteration
        n, res = 0, np.inf
        while n < self.newton_maxiter:
            # evaluate RHS
            f = self.dtype_u(u)
            f[0] = self.f(*u)
            f[1] = 1

            # form the function g with g(u) = 0
            g = u - dt * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble (dg/du)^{-1}
            dgInv = self.dgInv(u[0], u[1], dt)
            # newton update: u1 = u0 - g/dg
            u -= dgInv @ g

            # increase iteration count and work counter
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):  # pragma: no cover
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u


class Kaps(Problem):
    r"""
    Implement the Kaps problem:

    .. math::
        \begin{eqnarray*}
            \frac{du}{dt} &=& -(2+\epsilon^{-1})u + \frac{v^2}{\epsilon}, &\quad u(0) = 1,\\
            \frac{dv}{dt} &=& u - v(1+v), &\quad v(0) = 1,
        \end{eqnarray*}

    with :math:`\epsilon` a stiffness parameter, that makes the problem more stiff
    the smaller it is (usual taken value is :math:`\epsilon=1e^{-3}`).
    Exact solution is given by :math:`u(t)=e^{-2t},\;v(t)=e^{-t}`.

    Parameters
    ----------
    epsilon : float, optional
        Stiffness parameter. The default is 1e-3.
    newton_maxiter : int, optional
        Maximum number of Newton iteration in solve_system. The default is 200.
    newton_tol : float, optional
        Residuum tolerance for Newton iteration in solve_system. The default is 5e-11.
    stop_at_nan : bool, optional
        Wheter to stop or not solve_system when getting NAN. The default is True.

    Reference
    ---------
    Van der Houwen, P. J., & Sommeijer, B. P. (1991). Iterated Runge–Kutta methods
    on parallel computers. SIAM journal on scientific and statistical computing,
    12(5), 1000-1028.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, epsilon=1e-3, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 2
        super().__init__((nvars, None, np.dtype('float64')))

        self._makeAttributeAndRegister(
            'epsilon', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to return initial conditions or exact solutions.

        Parameters
        ----------
        t : float
            Time at which the exact solution is computed.
        u_init : dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        u : dtype_u
            The exact solution.
        """
        u = self.dtype_u(self.init)
        u[:] = [np.exp(-2 * t), np.exp(-t)]
        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (one component).
        """
        f = self.dtype_f(self.init)
        eps = self.epsilon
        x, y = u

        f[:] = [-(2 + 1 / eps) * x + y**2 / eps, x - y * (1 + y)]
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        eps = self.epsilon

        # start newton iteration
        n, res = 0, np.inf
        while n < self.newton_maxiter:
            x, y = u
            f = np.array([-(2 + 1 / eps) * x + y**2 / eps, x - y * (1 + y)])

            # form the function g with g(u) = 0
            g = u - dt * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble (dg/du)^(-1)
            prefactor = 4 * dt**2 * eps * y + 2 * dt**2 * eps + dt**2 + 2 * dt * eps * y + 3 * dt * eps + dt + eps
            dgInv = (
                1
                / prefactor
                * np.array([[2 * dt * eps * y + dt * eps + eps, 2 * dt * y], [dt * eps, 2 * dt * eps + dt + eps]])
            )

            # newton update: u1 = u0 - g/dg
            u -= dgInv @ g

            # increase iteration count and work counter
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):  # pragma: no cover
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u


class ChemicalReaction3Var(Problem):
    r"""
    Chemical reaction with three components, modeled by the non-linear system:

    .. math::
        \frac{d{\bf u}}{dt} =
        \begin{pmatrix}
            0.013+1000u_3 & 0 & 0 \\
            0 & 2500u_3 0 \\
            0.013 & 0 & 1000u_1 + 2500u_2
        \end{pmatrix}
        {\bf u},

    with initial solution :math:`u(0)=(0.990731920827, 1.009264413846, -0.366532612659e-5)`.

    Parameters
    ----------
    newton_maxiter : int, optional
        Maximum number of Newton iteration in solve_system. The default is 200.
    newton_tol : float, optional
        Residuum tolerance for Newton iteration in solve_system. The default is 5e-11.
    stop_at_nan : bool, optional
        Wheter to stop or not solve_system when getting NAN. The default is True.

    Reference
    ---------
    Van der Houwen, P. J., & Sommeijer, B. P. (1991). Iterated Runge–Kutta methods
    on parallel computers. SIAM journal on scientific and statistical computing,
    12(5), 1000-1028.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 3
        u0 = (0.990731920827, 1.009264413846, -0.366532612659e-5)
        super().__init__((nvars, None, np.dtype('float64')))

        self._makeAttributeAndRegister(
            'u0', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to return initial conditions or to approximate exact solution using ``SciPy``.

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
                r"""
                Evaluate the right hand side, but switch the arguments for ``SciPy``.

                Args:
                    t (float): Time
                    u (numpy.ndarray): Solution at time t

                Returns:
                    (numpy.ndarray): Right hand side evaluation
                """
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = self.u0
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (one component).
        """
        f = self.dtype_f(self.init)
        c1, c2, c3 = u

        f[:] = -np.array([0.013 * c1 + 1000 * c3 * c1, 2500 * c3 * c2, 0.013 * c1 + 1000 * c1 * c3 + 2500 * c2 * c3])
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)

        # start newton iteration
        n, res = 0, np.inf
        while n < self.newton_maxiter:
            c1, c2, c3 = u
            f = -np.array([0.013 * c1 + 1000 * c3 * c1, 2500 * c3 * c2, 0.013 * c1 + 1000 * c1 * c3 + 2500 * c2 * c3])

            # form the function g with g(u) = 0
            g = u - dt * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble (dg/du)^(-1)
            dgInv = np.array(
                [
                    [
                        (
                            2500000000.0 * c1 * c3**2 * dt**3
                            + 32500.0 * c1 * c3 * dt**3
                            + 3500000.0 * c1 * c3 * dt**2
                            + 13.0 * c1 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000.0 * c2 * c3 * dt**2
                            + 32.5 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000.0 * c3**2 * dt**2
                            + 32.5 * c3 * dt**2
                            + 3500.0 * c3 * dt
                            + 0.013 * dt
                            + 1.0
                        )
                        / (
                            2500000000.0 * c1 * c3**2 * dt**3
                            + 32500.0 * c1 * c3 * dt**3
                            + 3500000.0 * c1 * c3 * dt**2
                            + 13.0 * c1 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000000.0 * c2 * c3**2 * dt**3
                            + 65000.0 * c2 * c3 * dt**3
                            + 5000000.0 * c2 * c3 * dt**2
                            + 0.4225 * c2 * dt**3
                            + 65.0 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000000.0 * c3**3 * dt**3
                            + 65000.0 * c3**2 * dt**3
                            + 6000000.0 * c3**2 * dt**2
                            + 0.4225 * c3 * dt**3
                            + 91.0 * c3 * dt**2
                            + 4500.0 * c3 * dt
                            + 0.000169 * dt**2
                            + 0.026 * dt
                            + 1.0
                        ),
                        (2500000000.0 * c1 * c3**2 * dt**3 + 32500.0 * c1 * c3 * dt**3 + 2500000.0 * c1 * c3 * dt**2)
                        / (
                            2500000000.0 * c1 * c3**2 * dt**3
                            + 32500.0 * c1 * c3 * dt**3
                            + 3500000.0 * c1 * c3 * dt**2
                            + 13.0 * c1 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000000.0 * c2 * c3**2 * dt**3
                            + 65000.0 * c2 * c3 * dt**3
                            + 5000000.0 * c2 * c3 * dt**2
                            + 0.4225 * c2 * dt**3
                            + 65.0 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000000.0 * c3**3 * dt**3
                            + 65000.0 * c3**2 * dt**3
                            + 6000000.0 * c3**2 * dt**2
                            + 0.4225 * c3 * dt**3
                            + 91.0 * c3 * dt**2
                            + 4500.0 * c3 * dt
                            + 0.000169 * dt**2
                            + 0.026 * dt
                            + 1.0
                        ),
                        (
                            -2500000000.0 * c1 * c3**2 * dt**3
                            - 32500.0 * c1 * c3 * dt**3
                            - 3500000.0 * c1 * c3 * dt**2
                            - 13.0 * c1 * dt**2
                            - 1000.0 * c1 * dt
                        )
                        / (
                            2500000000.0 * c1 * c3**2 * dt**3
                            + 32500.0 * c1 * c3 * dt**3
                            + 3500000.0 * c1 * c3 * dt**2
                            + 13.0 * c1 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000000.0 * c2 * c3**2 * dt**3
                            + 65000.0 * c2 * c3 * dt**3
                            + 5000000.0 * c2 * c3 * dt**2
                            + 0.4225 * c2 * dt**3
                            + 65.0 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000000.0 * c3**3 * dt**3
                            + 65000.0 * c3**2 * dt**3
                            + 6000000.0 * c3**2 * dt**2
                            + 0.4225 * c3 * dt**3
                            + 91.0 * c3 * dt**2
                            + 4500.0 * c3 * dt
                            + 0.000169 * dt**2
                            + 0.026 * dt
                            + 1.0
                        ),
                    ],
                    [
                        (6250000000.0 * c2 * c3 * dt**2 + 81250.0 * c2 * dt**2)
                        / (
                            6250000000.0 * c1 * c3 * dt**2
                            + 2500000.0 * c1 * dt
                            + 6250000000.0 * c2 * c3 * dt**2
                            + 81250.0 * c2 * dt**2
                            + 6250000.0 * c2 * dt
                            + 6250000000.0 * c3**2 * dt**2
                            + 81250.0 * c3 * dt**2
                            + 8750000.0 * c3 * dt
                            + 32.5 * dt
                            + 2500.0
                        ),
                        (
                            2500000.0 * c1 * dt
                            + 6250000000.0 * c2 * c3 * dt**2
                            + 81250.0 * c2 * dt**2
                            + 6250000.0 * c2 * dt
                            + 2500000.0 * c3 * dt
                            + 32.5 * dt
                            + 2500.0
                        )
                        / (
                            6250000000.0 * c1 * c3 * dt**2
                            + 2500000.0 * c1 * dt
                            + 6250000000.0 * c2 * c3 * dt**2
                            + 81250.0 * c2 * dt**2
                            + 6250000.0 * c2 * dt
                            + 6250000000.0 * c3**2 * dt**2
                            + 81250.0 * c3 * dt**2
                            + 8750000.0 * c3 * dt
                            + 32.5 * dt
                            + 2500.0
                        ),
                        (-6250000000.0 * c2 * c3 * dt**2 - 81250.0 * c2 * dt**2 - 6250000.0 * c2 * dt)
                        / (
                            6250000000.0 * c1 * c3 * dt**2
                            + 2500000.0 * c1 * dt
                            + 6250000000.0 * c2 * c3 * dt**2
                            + 81250.0 * c2 * dt**2
                            + 6250000.0 * c2 * dt
                            + 6250000000.0 * c3**2 * dt**2
                            + 81250.0 * c3 * dt**2
                            + 8750000.0 * c3 * dt
                            + 32.5 * dt
                            + 2500.0
                        ),
                    ],
                    [
                        (-2500000.0 * c3**2 * dt**2 - 32.5 * c3 * dt**2 - 1000.0 * c3 * dt - 0.013 * dt)
                        / (
                            2500000.0 * c1 * c3 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000.0 * c2 * c3 * dt**2
                            + 32.5 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000.0 * c3**2 * dt**2
                            + 32.5 * c3 * dt**2
                            + 3500.0 * c3 * dt
                            + 0.013 * dt
                            + 1.0
                        ),
                        (-2500000.0 * c3**2 * dt**2 - 32.5 * c3 * dt**2 - 2500.0 * c3 * dt)
                        / (
                            2500000.0 * c1 * c3 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000.0 * c2 * c3 * dt**2
                            + 32.5 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000.0 * c3**2 * dt**2
                            + 32.5 * c3 * dt**2
                            + 3500.0 * c3 * dt
                            + 0.013 * dt
                            + 1.0
                        ),
                        (2500000.0 * c3**2 * dt**2 + 32.5 * c3 * dt**2 + 3500.0 * c3 * dt + 0.013 * dt + 1.0)
                        / (
                            2500000.0 * c1 * c3 * dt**2
                            + 1000.0 * c1 * dt
                            + 2500000.0 * c2 * c3 * dt**2
                            + 32.5 * c2 * dt**2
                            + 2500.0 * c2 * dt
                            + 2500000.0 * c3**2 * dt**2
                            + 32.5 * c3 * dt**2
                            + 3500.0 * c3 * dt
                            + 0.013 * dt
                            + 1.0
                        ),
                    ],
                ]
            )

            # newton update: u1 = u0 - g/dg
            u -= dgInv @ g

            # increase iteration count and work counter
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):  # pragma: no cover
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u


class JacobiElliptic(Problem):
    r"""
    Implement the Jacobi Elliptic non-linear problem:

    .. math::
        \begin{eqnarray*}
            \frac{du}{dt} &=& vw, &\quad u(0) = 0,      \\
            \frac{dv}{dt} &=& -uw, &\quad v(0) = 1,     \\
            \frac{dw}{dt} &=& -0.51uv, &\quad w(0) = 1.
        \end{eqnarray*}

    Parameters
    ----------
    newton_maxiter : int, optional
        Maximum number of Newton iteration in solve_system. The default is 200.
    newton_tol : float, optional
        Residuum tolerance for Newton iteration in solve_system. The default is 5e-11.
    stop_at_nan : bool, optional
        Wheter to stop or not solve_system when getting NAN. The default is True.

    Reference
    ---------
    Van Der Houwen, P. J., Sommeijer, B. P., & Van Der Veen, W. A. (1995).
    Parallel iteration across the steps of high-order Runge-Kutta methods for
    nonstiff initial value problems. Journal of computational and applied
    mathematics, 60(3), 309-329.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 3
        u0 = (0.0, 1.0, 1.0)
        super().__init__((nvars, None, np.dtype('float64')))

        self._makeAttributeAndRegister(
            'u0', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to return initial conditions or to approximate exact solution using ``SciPy``.

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
                r"""
                Evaluate the right hand side, but switch the arguments for ``SciPy``.

                Args:
                    t (float): Time
                    u (numpy.ndarray): Solution at time t

                Returns:
                    (numpy.ndarray): Right hand side evaluation
                """
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = self.u0
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (one component).
        """
        f = self.dtype_f(self.init)
        u1, u2, u3 = u

        f[:] = np.array([u2 * u3, -u1 * u3, -0.51 * u1 * u2])
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)

        # start newton iteration
        n, res = 0, np.inf
        while n < self.newton_maxiter:
            u1, u2, u3 = u
            f = np.array([u2 * u3, -u1 * u3, -0.51 * u1 * u2])

            # form the function g with g(u) = 0
            g = u - dt * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble (dg/du)^(-1)
            dgInv = np.array(
                [
                    [
                        0.51 * dt**2 * u1**2 - 1.0,
                        0.51 * dt**2 * u1 * u2 - 1.0 * dt * u3,
                        1.0 * dt**2 * u1 * u3 - 1.0 * dt * u2,
                    ],
                    [
                        -0.51 * dt**2 * u1 * u2 + 1.0 * dt * u3,
                        -0.51 * dt**2 * u2**2 - 1.0,
                        1.0 * dt**2 * u2 * u3 + 1.0 * dt * u1,
                    ],
                    [
                        -0.51 * dt**2 * u1 * u3 + 0.51 * dt * u2,
                        0.51 * dt**2 * u2 * u3 + 0.51 * dt * u1,
                        -1.0 * dt**2 * u3**2 - 1.0,
                    ],
                ]
            )
            dgInv /= (
                1.02 * dt**3 * u1 * u2 * u3 + 0.51 * dt**2 * u1**2 - 0.51 * dt**2 * u2**2 - 1.0 * dt**2 * u3**2 - 1.0
            )

            # newton update: u1 = u0 - g/dg
            u -= dgInv @ g

            # increase iteration count and work counter
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):  # pragma: no cover
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u
