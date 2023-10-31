import numpy as np
from scipy.optimize import root

from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


class ptype_dae(ptype):
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

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """Initialization routine"""
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def solve_system(self, impl_sys, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
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
        opt = root(
            impl_sys,
            u0,
            method='hybr',
            tol=self.newton_tol,
        )
        me[:] = opt.x
        self.work_counters['newton'].niter += opt.nfev
        return me
