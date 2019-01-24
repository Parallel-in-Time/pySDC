import numpy as np
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.problem_classes.acoustic_helpers.buildWave1DMatrix import getWave1DMatrix, \
    getWave1DAdvectionMatrix


# noinspection PyUnusedLocal
class acoustic_1d_imex(ptype):
    """
    Example implementing the one-dimensional IMEX acoustic-advection

    Attributes:
        mesh (numpy.ndarray): 1d mesh
        dx (float): mesh size
        Dx: matrix for the advection operator
        Id: sparse identity matrix
        A: matrix for the wave operator

    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=rhs_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data
            dtype_f: mesh data with two components
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'cs', 'cadv', 'order_adv', 'waveno']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(acoustic_1d_imex, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        self.mesh = np.linspace(0.0, 1.0, self.params.nvars[1], endpoint=False)
        self.dx = self.mesh[1] - self.mesh[0]

        self.Dx = -self.params.cadv * getWave1DAdvectionMatrix(self.params.nvars[1], self.dx, self.params.order_adv)
        self.Id, A = getWave1DMatrix(self.params.nvars[1], self.dx, ['periodic', 'periodic'], ['periodic', 'periodic'])
        self.A = -self.params.cs * A

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        M = self.Id - factor * self.A

        b = np.concatenate((rhs.values[0, :], rhs.values[1, :]))

        sol = spsolve(M, b)

        me = self.dtype_u(self.init)
        me.values[0, :], me.values[1, :] = np.split(sol, 2)

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            explicit part of RHS
        """

        b = np.concatenate((u.values[0, :], u.values[1, :]))
        sol = self.Dx.dot(b)

        fexpl = self.dtype_u(self.init)
        fexpl.values[0, :], fexpl.values[1, :] = np.split(sol, 2)

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            implicit part of RHS
        """

        b = np.concatenate((u.values[0, :], u.values[1, :]))
        sol = self.A.dot(b)

        fimpl = self.dtype_u(self.init, val=0)
        fimpl.values[0, :], fimpl.values[1, :] = np.split(sol, 2)

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        def u_initial(x, k):
            return np.sin(k * 2.0 * np.pi * x) + np.sin(2.0 * np.pi * x)

        me = self.dtype_u(self.init)
        me.values[0, :] = 0.5 * u_initial(self.mesh - (self.params.cadv + self.params.cs) * t, self.params.waveno) - \
            0.5 * u_initial(self.mesh - (self.params.cadv - self.params.cs) * t, self.params.waveno)
        me.values[1, :] = 0.5 * u_initial(self.mesh - (self.params.cadv + self.params.cs) * t, self.params.waveno) + \
            0.5 * u_initial(self.mesh - (self.params.cadv - self.params.cs) * t, self.params.waveno)
        return me
