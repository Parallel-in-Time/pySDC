import numpy as np
from scipy.optimize import root

from pySDC.core.problem import Problem, WorkCounter
from pySDC.projects.DAE.misc.meshDAE import MeshDAE


class ProblemDAE(Problem):
    r"""
    This class implements a generic DAE class and illustrates the interface class for DAE problems.
    It ensures that all parameters are passed that are needed by DAE sweepers.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the problem class.
    newton_tol : float
        Tolerance for the nonlinear solver.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts the work, here the number of function calls during the nonlinear solve is logged and stored
        in work_counters['newton']. The number of each function class of the right-hand side is then stored
        in work_counters['rhs']
    """

    dtype_u = MeshDAE
    dtype_f = MeshDAE

    def __init__(self, nvars, newton_tol):
        """Initialization routine"""
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u_approx : dtype_u
            Approximation of solution :math:`u` which is needed to solve
            the implicit system.
        factor : float
            Abbrev. for the node-to-node stepsize.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """
        me = self.dtype_u(self.init)

        def implSysFlatten(unknowns, **kwargs):
            sys = impl_sys(unknowns.reshape(me.shape).view(type(u0)), self, factor, u_approx, t, **kwargs)
            return sys.flatten()

        opt = root(
            implSysFlatten,
            u0,
            method='hybr',
            tol=self.newton_tol,
        )
        me[:] = opt.x.reshape(me.shape)
        self.work_counters['newton'].niter += opt.nfev
        return me

    def du_exact(self, t):
        r"""
        Routine for the derivative of the exact solution at time :math:`t \leq 1`.
        For this problem, the exact solution is piecewise.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Derivative of exact solution.
        """

        raise NotImplementedError('ERROR: problem has to implement du_exact(self, t)!')
