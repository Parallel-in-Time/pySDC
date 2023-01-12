import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery_2condensators(ptype):
    """
    Example implementing the battery drain model using two capacitors as in the description in the PinTSimE
    project
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
        essential_keys = ['Vs', 'Rs', 'C1', 'C2', 'R', 'L', 'alpha', 'V_ref']

        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_2condensators, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((3, 3))
        self.t_switch = None

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if u[2] > self.params.V_ref[1]:
            if t >= self.t_switch if self.t_switch is not None else u[1] <= self.params.V_ref[0]:
                # switch to C2
                f.expl[0] = 0

            elif u[1] > self.params.V_ref[0]:
                # C1 supplies energy
                f.expl[0] = 0

        elif t >= self.t_switch if self.t_switch is not None else u[2] <= self.params.V_ref[1]:
            # switch to Vs
            f.expl[0] = self.params.Vs / self.params.L

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
        self.A = np.zeros((3, 3))

        if rhs[2] > self.params.V_ref[1]:
            if t >= self.t_switch if self.t_switch is not None else rhs[1] <= self.params.V_ref[0]:
                # switch to C2
                self.A[2, 2] = -1 / (self.params.C2 * self.params.R)

            elif rhs[1] > self.params.V_ref[0]:
                # C1 supplies energy
                self.A[1, 1] = -1 / (self.params.C1 * self.params.R)

        elif t >= self.t_switch if self.t_switch is not None else rhs[2] <= self.params.V_ref[1]:
            # switch to Vs
            self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L

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

        me[0] = 0.0  # cL
        me[1] = self.params.alpha * self.params.V_ref[0]  # vC1
        me[2] = self.params.alpha * self.params.V_ref[1]  # vC2
        return me
