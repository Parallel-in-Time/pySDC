from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit_jac(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_fullyimplicit_jac, self).__init__(problem_params, dtype_u, dtype_f)

        self.Jf = None

        self.inner_solve_counter = 0

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        me = self.dtype_u(self.init)

        me.values = spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.Jf, rhs.values)

        self.inner_solve_counter += 1

        return me

    def eval_f_ode(self, u, t):
        """
        Routine to evaluate the RHS of the ODE

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)

        f.values = self.A.dot(u.values) + 1.0 / self.params.eps ** 2 * u.values * (1.0 - u.values ** self.params.nu)

        return f

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS of the ODE

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)

        f.values = self.Jf.dot(u.values)

        return f

    def build_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        # noinspection PyTypeChecker
        self.Jf = self.A + \
                  sp.diags(1.0 / self.params.eps ** 2 * (1.0 - (self.params.nu + 1) * u.values ** self.params.nu),
                           offsets=0)
