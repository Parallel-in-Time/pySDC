import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, parallel_imex_mesh
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d


# noinspection PyUnusedLocal
class heat1d_forced(heat1d):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1],
    discretized using central finite differences
    """
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=parallel_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d_forced, self).__init__(problem_params, dtype_u, dtype_f)

        self.xvalues = np.array([(i + 1) * self.dx for i in range(self.params.nvars)])

        pass

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            dtype_f: explicit part of RHS
        """

        fexpl = self.dtype_u(self.init)
        fexpl[:] = -np.sin(np.pi * self.params.freq * self.xvalues) * \
            (np.sin(t) - self.params.nu * (np.pi * self.params.freq) ** 2 * np.cos(t))
        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            dtype_f: implicit part of RHS
        """

        fimpl = self.dtype_u(self.init)
        fimpl[:] = self.A.dot(u)
        return fimpl

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        me[:] = np.sin(np.pi * self.params.freq * self.xvalues) * np.cos(t)
        return me
