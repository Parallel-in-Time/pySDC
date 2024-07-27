import numpy as np

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.implementations.datatype_classes.mesh import mesh


class ProblematicF(ProblemDAE):
    r"""
    Standard example of a very simple fully implicit index-2 differential algebraic equation (DAE) that is not
    numerically solvable for certain choices of the parameter :math:`\eta`. The DAE system is given by

    .. math::
        \frac{d y(t)}{dt} + \eta t \frac{d z(t)}{dt} + (1 + \eta) z (t) = g (t).

    .. math::
        y (t) + \eta t z (t) = f(t),

    See, for example, page 264 of [1]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    eta : float
        Specific parameter of the problem.

    References
    ----------
    .. [1] U. Ascher, L. R. Petzold. Computer method for ordinary differential equations and differential-algebraic
        equations. Society for Industrial and Applied Mathematics (1998).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_tol, eta=1):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister('eta', localVars=locals())

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            Current value of the right-hand side of f (which includes two components).
        """
        f = self.dtype_f(self.init)
        f[:] = (
            u[0] + self.eta * t * u[1] - np.sin(t),
            du[0] + self.eta * t * du[1] + (1 + self.eta) * u[1] - np.cos(t),
        )
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Routine for the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing two components.
        """
        me = self.dtype_u(self.init)
        me[:] = (np.sin(t), 0)
        return me

    def du_exact(self, t):
        """
        Routine for the derivative of the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing two components.
        """

        me = self.dtype_u(self.init)
        me[:] = (np.cos(t), 0)
        return me
