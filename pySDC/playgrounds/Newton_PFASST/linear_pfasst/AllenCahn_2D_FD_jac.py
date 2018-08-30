from __future__ import division

import time
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit_jac(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_fullyimplicit_jac, self).__init__(problem_params, dtype_u, dtype_f)

        self.Jf = None

        self.inner_solve_counter = 0

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):

        num_iters = 0

        def callback(xk):
            nonlocal num_iters
            num_iters += 1

        t0 = time.time()
        me = self.dtype_u(self.init)
        z = self.dtype_u(self.init, val=0.0)

        M = sp.eye(self.params.nvars[0] * self.params.nvars[1], format='csr') - factor * self.Jf

        # me.values = spsolve(sp.eye(self.params.nvars[0] * self.params.nvars[1], format='csc') - factor * self.Jf, rhs.values.flatten())
        me.values = cg(M, rhs.values.flatten(), x0=z.values.flatten(), tol=self.params.lin_tol, callback=callback)[0]
        me.values = me.values.reshape(self.params.nvars)
        print('.......... %s -- %s' % (time.time() - t0, num_iters))

        self.inner_solve_counter += 1

        return me

    def eval_f_ode(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        v = u.values.flatten()
        f.values = self.A.dot(v) + 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
        f.values = f.values.reshape(self.params.nvars)

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

        f.values = self.Jf.dot(u.values.flatten())
        f.values = f.values.reshape(self.params.nvars)

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
        self.Jf = self.A + sp.diags(1.0 / self.params.eps ** 2 * (1.0 - (self.params.nu + 1) * u.values.flatten() ** self.params.nu),
                                    offsets=0)

