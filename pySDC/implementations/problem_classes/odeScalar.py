#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of scalar test problem ODEs.


Reference :

Van der Houwen, P. J., & Sommeijer, B. P. (1991). Iterated Runge–Kutta methods
on parallel computers. SIAM journal on scientific and statistical computing,
12(5), 1000-1028.
"""
import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


class ProtheroRobinson(Problem):
    r"""
    Implement the Prothero-Robinson problem:

    .. math::
        \frac{du}{dt} = -\frac{u-g(t)}{\epsilon} + \frac{dg}{dt}, \quad u(0) = g(0).,

    with :math:`\epsilon` a stiffness parameter, that makes the problem more stiff
    the smaller it is (usual taken value is :math:`\epsilon=1e^{-3}`).
    Exact solution is given by :math:`u(t)=g(t)`, and this implementation uses
    :math:`g(t)=\cos(t)`.

    Implement also the non-linear form of this problem:

    .. math::
        \frac{du}{dt} = -\frac{u^3-g(t)^3}{\epsilon} + \frac{dg}{dt}, \quad u(0) = g(0).

    To use an other exact solution, one just have to derivate this class
    and overload the `g` and `dg` methods. For instance,
    to use :math:`g(t)=e^{-0.2*t}`, define and use the following class:

    >>> class MyProtheroRobinson(ProtheroRobinson):
    >>>
    >>>     def g(self, t):
    >>>         return np.exp(-0.2 * t)
    >>>
    >>>     def dg(self, t):
    >>>         return (-0.2) * np.exp(-0.2 * t)

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
        nvars = 1
        super().__init__((nvars, None, np.dtype('float64')))

        self.f = self.f_NONLIN if nonLinear else self.f_LIN
        self.jac = self.jac_NONLIN if nonLinear else self.jac_LIN
        self._makeAttributeAndRegister(
            'epsilon', 'nonLinear', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    # -------------------------------------------------------------------------
    # g function (analytical solution), and its first derivative
    # -------------------------------------------------------------------------
    def g(self, t):
        return np.cos(t)

    def dg(self, t):
        return -np.sin(t)

    # -------------------------------------------------------------------------
    # f(u,t) and Jacobian functions
    # -------------------------------------------------------------------------
    def f(self, u, t):
        raise NotImplementedError()

    def f_LIN(self, u, t):
        return -self.epsilon ** (-1) * (u - self.g(t)) + self.dg(t)

    def f_NONLIN(self, u, t):
        return -self.epsilon ** (-1) * (u**3 - self.g(t) ** 3) + self.dg(t)

    def jac(self, u, t):
        raise NotImplementedError()

    def jac_LIN(self, u, t):
        return -self.epsilon ** (-1)

    def jac_NONLIN(self, u, t):
        return -self.epsilon ** (-1) * 3 * u**2

    # -------------------------------------------------------------------------
    # pySDC required methods
    # -------------------------------------------------------------------------
    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to return initial conditions or exact solution.

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
        u[:] = self.g(t)
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
        f[:] = self.f(u, t)
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
            # form the function g with g(u) = 0
            g = u - dt * self.f(u, t) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble dg/du
            dg = 1 - dt * self.jac(u, t)

            # newton update: u1 = u0 - g/dg
            u -= dg ** (-1) * g

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
