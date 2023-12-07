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

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


class Kaps(ptype):

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, epsilon=1e-3, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 2
        super().__init__((nvars, None, np.dtype('float64')))

        self._makeAttributeAndRegister(
            'epsilon', 'newton_maxiter', 'newton_tol', 'stop_at_nan',
            localVars=locals(), readOnly=True
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
        u = self.dtype_u(self.init)
        u[:] = [np.exp(-2*t), np.exp(-t)]
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

        f[:] = [-(2+1/eps)*x + y**2/eps, x-y*(1+y)]
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
            f = np.array([-(2+1/eps)*x + y**2/eps, x-y*(1+y)])

            # form the function g with g(u) = 0
            g = u - dt * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble (dg/du)^(-1)
            prefactor = 4*dt**2*eps*y + 2*dt**2*eps + dt**2 + 2*dt*eps*y + 3*dt*eps + dt + eps
            dgInv = 1/prefactor * np.array([
                [2*dt*eps*y + dt*eps + eps, 2*dt*y],
                [dt*eps, 2*dt*eps + dt + eps]
                ])

            # newton update: u1 = u0 - g/dg
            u -= dgInv @ g

            # increase iteration count and work counter
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u
