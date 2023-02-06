import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class buck_converter(ptype):
    """
    Example implementing the buck converter model as in the description in the PinTSimE project
    Attributes:
        A: system matrix, representing the 3 ODEs
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 3

        # these parameters will be used later, so assert their existence
        essential_keys = ['duty', 'fsw', 'Vs', 'Rs', 'C1', 'Rp', 'L1', 'C2', 'Rl']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(buck_converter, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((3, 3))

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """
        Tsw = 1 / self.params.fsw

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if 0 <= ((t / Tsw) % 1) <= self.params.duty:
            f.expl[0] = self.params.Vs / (self.params.Rs * self.params.C1)
            f.expl[2] = 0

        else:
            f.expl[0] = self.params.Vs / (self.params.Rs * self.params.C1)
            f.expl[2] = -(self.params.Rp * self.params.Vs) / (self.params.L1 * self.params.Rs)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs
        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)
        Returns:
            dtype_u: solution as mesh
        """
        Tsw = 1 / self.params.fsw
        self.A = np.zeros((3, 3))

        if 0 <= ((t / Tsw) % 1) <= self.params.duty:
            self.A[0, 0] = -1 / (self.params.C1 * self.params.Rs)
            self.A[0, 2] = -1 / self.params.C1

            self.A[1, 1] = -1 / (self.params.C2 * self.params.Rl)
            self.A[1, 2] = 1 / self.params.C2

            self.A[2, 0] = 1 / self.params.L1
            self.A[2, 1] = -1 / self.params.L1
            self.A[2, 2] = -self.params.Rp / self.params.L1

        else:
            self.A[0, 0] = -1 / (self.params.C1 * self.params.Rs)

            self.A[1, 1] = -1 / (self.params.C2 * self.params.Rl)
            self.A[1, 2] = 1 / self.params.C2

            self.A[2, 0] = self.params.Rp / (self.params.L1 * self.params.Rs)
            self.A[2, 1] = -1 / self.params.L1

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t
        Args:
            t (float): current time
        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # v1
        me[1] = 0.0  # v2
        me[2] = 0.0  # p3

        return me
