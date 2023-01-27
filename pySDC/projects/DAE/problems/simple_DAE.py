import numpy as np

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class pendulum_2d(ptype_dae):
    """
    Example implementing the well known 2D pendulum as a first order DAE of index-3
    The pendulum is used in most introductory literature on DAEs, for example on page 8 of "The numerical solution of differential-algebraic systems by Runge-Kutta methods" by Hairer et al.
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine for the problem class
        """
        super(pendulum_2d, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time (not used here)
        Returns:
            Current value of F(), 5 components
        """
        g = 9.8
        # The last element of u is a Lagrange multiplier. Not sure if this needs to be time dependent, but must model the
        # weight somehow
        f = self.dtype_f(self.init)
        f[:] = (du[0] - u[2], du[1] - u[3], du[2] + u[4] * u[0], du[3] + u[4] * u[1] + g, u[0] ** 2 + u[1] ** 2 - 1)
        return f

    def u_exact(self, t):
        """
        Dummy exact solution that should only be used to get initial conditions for the problem
        This makes initialisation compatible with problems that have a known analytical solution
        Could be used to output a reference solution if generated/available
        Args:
            t (float): current time (not used here)
        Returns:
            Mesh containing fixed initial value, 5 components
        """
        me = self.dtype_u(self.init)
        me[:] = (-1, 0, 0, 0, 0)
        return me


class simple_dae_1(ptype_dae):
    """
    Example implementing a smooth linear index-2 DAE with known analytical solution
    This example is commonly used to test that numerical implementations are functioning correctly
    See, for example, page 267 of "computer methods for ODEs and DAEs" by Ascher and Petzold
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine for the problem class
        """
        super(simple_dae_1, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 3 components
        """
        # Smooth index-2 DAE pg. 267 Ascher and Petzold (also the first example in KDC Minion paper)
        a = 10.0
        f = self.dtype_f(self.init)
        f[:] = (
            -du[0] + (a - 1 / (2 - t)) * u[0] + (2 - t) * a * u[2] + np.exp(t) * (3 - t) / (2 - t),
            -du[1] + (1 - a) / (t - 2) * u[0] - u[1] + (a - 1) * u[2] + 2 * np.exp(t),
            (t + 2) * u[0] + (t**2 - 4) * u[1] - (t**2 + t - 2) * np.exp(t),
        )
        return f

    def u_exact(self, t):
        """
        Routine for the exact solution

        Args:
            t (float): current time
        Returns:
            mesh type containing the exact solution, 3 components
        """
        me = self.dtype_u(self.init)
        me[:] = (np.exp(t), np.exp(t), -np.exp(t) / (2 - t))
        return me


class problematic_f(ptype_dae):
    """
    Standard example of a very simple fully implicit index-2 DAE that is not numerically solvable for certain choices of the parameter eta
    See, for example, page 264 of "computer methods for ODEs and DAEs" by Ascher and Petzold
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh, eta=1):
        """
        Initialization routine for the problem class
        """
        self.eta = eta
        super().__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 2 components
        """
        f = self.dtype_f(self.init)
        f[:] = (
            u[0] + self.eta * t * u[1] - np.sin(t),
            du[0] + self.eta * t * du[1] + (1 + self.eta) * u[1] - np.cos(t),
        )
        return f

    def u_exact(self, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 2 components
        """
        me = self.dtype_u(self.init)
        me[:] = (np.sin(t), 0)
        return me
