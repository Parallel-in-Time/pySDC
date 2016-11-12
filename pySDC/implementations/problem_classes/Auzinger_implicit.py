from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype


class auzinger(ptype):
    """
    Example implementing the Auzinger initial value problem
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'newton_maxiter' in cparams
        assert 'newton_tol' in cparams

        # add parameters as attributes for further reference
        for k, v in cparams.items():
            setattr(self, k, v)
        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super(auzinger, self).__init__(2, dtype_u, dtype_f, cparams)

    def u_exact(self, t):
        """
        Routine for the exact solution

        Args:
            t: current time
        Returns:
            mesh type containing the exact solution
        """

        me = self.dtype_u(self.init)
        me.values[0] = np.cos(t)
        me.values[1] = np.sin(t)
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the RHS for both components simultaneously

        Args:
            t: current time (not used here)
            u: the current values
        Returns:
            RHS, 2 components
        """

        x1 = u.values[0]
        x2 = u.values[1]
        f = self.dtype_f(self.init)
        f.values[0] = -x2 + x1 * (1 - x1 ** 2 - x2 ** 2)
        f.values[1] = x1 + 3 * x2 * (1 - x1 ** 2 - x2 ** 2)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs: right-hand side for the nonlinear system
            dt: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution u
        """

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = u.values[0]
        x2 = u.values[1]

        # start newton iteration
        n = 0
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = np.array([x1 - dt * (-x2 + x1 * (1 - x1 ** 2 - x2 ** 2)) - rhs.values[0],
                          x2 - dt * (x1 + 3 * x2 * (1 - x1 ** 2 - x2 ** 2)) - rhs.values[1]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg and invert the matrix (yeah, I know)
            dg = np.array([[1 - dt * (1 - 3 * x1 ** 2 - x2 ** 2), -dt * (-1 - 2 * x1 * x2)],
                           [-dt * (1 - 6 * x1 * x2), 1 - dt * (3 - 3 * x1 ** 2 - 9 * x2 ** 2)]])

            idg = np.linalg.inv(dg)

            # newton update: u1 = u0 - g/dg
            u.values -= np.dot(idg, g)

            # set new values and increase iteration count
            x1 = u.values[0]
            x2 = u.values[1]
            n += 1

        return u

        # def eval_jacobian(self, u):
        #
        #     x1 = u.values[0]
        #     x2 = u.values[1]
        #
        #     dfdu = np.array([[1-3*x1**2-x2**2, -1-x1], [1+6*x2*x1, 3+3*x1**2-9*x2**2]])
        #
        #     return dfdu
        #
        #
        # def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        #     """
        #     Simple linear solver for (I-dtA)u = rhs
        #
        #     Args:
        #         dfdu: the Jacobian of the RHS of the ODE
        #         rhs: right-hand side for the linear system
        #         factor: abbrev. for the node-to-node stepsize (or any other factor required)
        #         u0: initial guess for the iterative solver (not used here so far)
        #         t: current time (e.g. for time-dependent BCs)
        #
        #     Returns:
        #         solution as mesh
        #     """
        #
        #     me = mesh(2)
        #     me.values = LA.spsolve(sp.eye(2) - factor * dfdu, rhs.values)
        #     return me
