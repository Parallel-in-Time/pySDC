import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class auzinger(ptype):
    """
    This class implements the Auzinger equation as initial value problem. It can be found in doi.org/10.2140/camcos.2015.10.1.
    The system of two ordinary differential equations (ODEs) is given by

    .. math::
        \frac{d y_1 (t)}{dt} = -y_2 (t) + y_1 (t) (1 - y^2_1 (t) - y^2_2 (t)),

    .. math::
        \frac{d y_2 (t)}{dt} = y_1 (t) + 3 y_2 (t) (1 - y^2_1 (t) - y^2_2 (t))

    with initial condition :math:`y(t) = (1, 0)^T` for :math:`t \in [0, 10]`. The exact solution of this problem is

    .. math::
        y (t) = (\cos(t), \sin(t))^T.

    Attributes
    ----------
    newton_maxiter : int, optional
        Maximum number of iterations for Newton's method.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_maxiter=1e-12, newton_tol=100):
        """Initialization routine"""

        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super().__init__((2, None, np.dtype('float64')))
        self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals(), readOnly=True)

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init)
        me[0] = np.cos(t)
        me[1] = np.sin(t)
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side of the problem for both components simultaneously.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (contains two components).
        """

        x1 = u[0]
        x2 = u[1]
        f = self.dtype_f(self.init)
        f[0] = -x2 + x1 * (1 - x1**2 - x2**2)
        f[1] = x1 + 3 * x2 * (1 - x1**2 - x2**2)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs.)

        Returns
        -------
        u : dtype_u
            The solution as mesh.
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
