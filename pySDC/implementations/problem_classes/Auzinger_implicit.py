import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class auzinger(ptype):
    """
    Example implementing the Auzinger initial value problem
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_maxiter, newton_tol):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type (will be passed to parent class)
        """
        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super().__init__((2, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_maxiter', 'newton_tol', localVars=locals(), readOnly=True)

    def u_exact(self, t):
        """
        Routine for the exact solution

        Args:
            t (float): current time
        Returns:
            dtype_u: mesh type containing the exact solution
        """

        me = self.dtype_u(self.init)
        me[0] = np.cos(t)
        me[1] = np.sin(t)
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the RHS for both components simultaneously

        Args:
            u (dtype_u): the current values
            t (float): current time (not used here)
        Returns:
            RHS, 2 components
        """

        x1 = u[0]
        x2 = u[1]
        f = self.dtype_f(self.init)
        f[0] = -x2 + x1 * (1 - x1**2 - x2**2)
        f[1] = x1 + 3 * x2 * (1 - x1**2 - x2**2)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            dt (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution u
        """

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = u[0]
        x2 = u[1]

        # start newton iteration
        n = 0
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = np.array(
                [
                    x1 - dt * (-x2 + x1 * (1 - x1**2 - x2**2)) - rhs[0],
                    x2 - dt * (x1 + 3 * x2 * (1 - x1**2 - x2**2)) - rhs[1],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg and invert the matrix (yeah, I know)
            dg = np.array(
                [
                    [1 - dt * (1 - 3 * x1**2 - x2**2), -dt * (-1 - 2 * x1 * x2)],
                    [-dt * (1 - 6 * x1 * x2), 1 - dt * (3 - 3 * x1**2 - 9 * x2**2)],
                ]
            )

            idg = np.linalg.inv(dg)

            # newton update: u1 = u0 - g/dg
            u -= np.dot(idg, g)

            # set new values and increase iteration count
            x1 = u[0]
            x2 = u[1]
            n += 1

        return u

        # def eval_jacobian(self, u):
        #
        #     x1 = u[0]
        #     x2 = u[1]
        #
        #     dfdu = np.array([[1-3*x1**2-x2**2, -1-x1], [1+6*x2*x1, 3+3*x1**2-9*x2**2]])
        #
        #     return dfdu
        #
        #
        # def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        #
        #     me = mesh(2)
        #     me = LA.spsolve(sp.eye(2) - factor * dfdu, rhs)
        #     return me
