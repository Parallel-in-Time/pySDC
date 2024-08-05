import numpy as np

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class SimpleDAE(ProblemDAE):
    r"""
    Example implementing a smooth linear index-2 differential-algebraic equation (DAE) with known analytical solution.
    The DAE system is given by

    .. math::
        \frac{d u_1 (t)}{dt} = (\alpha - \frac{1}{2 - t}) u_1 (t) + (2-t) \alpha z (t) + \frac{3 - t}{2 - t},

    .. math::
        \frac{d u_2 (t)}{dt} = \frac{1 - \alpha}{t - 2} u_1 (t) - u_2 (t) + (\alpha - 1) z (t) + 2 e^{t},

    .. math::
        0 = (t + 2) u_1 (t) + (t^{2} - 4) u_2 (t) - (t^{2} + t - 2) e^{t}.

    The exact solution of this system is

    .. math::
        u_1 (t) = u_2 (t) = e^{t},

    .. math::
        z (t) = -\frac{e^{t}}{2 - t}.

    This example is commonly used to test that numerical implementations are functioning correctly. See, for example,
    page 267 of [1]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    References
    ----------
    .. [1] U. Ascher, L. R. Petzold. Computer method for ordinary differential equations and differential-algebraic
        equations. Society for Industrial and Applied Mathematics (1998).
    """

    def __init__(self, newton_tol=1e-10):
        """Initialization routine"""
        super().__init__(nvars=3, newton_tol=newton_tol)

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
            Current value of the right-hand side of f (which includes three components).
        """
        # Smooth index-2 DAE pg. 267 Ascher and Petzold (also the first example in KDC Minion paper)
        a = 10.0
        f = self.dtype_f(self.init)

        f.diff[:2] = (
            -du.diff[0] + (a - 1 / (2 - t)) * u.diff[0] + (2 - t) * a * u.alg[0] + (3 - t) / (2 - t) * np.exp(t),
            -du.diff[1] + (1 - a) / (t - 2) * u.diff[0] - u.diff[1] + (a - 1) * u.alg[0] + 2 * np.exp(t),
        )
        f.alg[0] = (t + 2) * u.diff[0] + (t**2 - 4) * u.diff[1] - (t**2 + t - 2) * np.exp(t)
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
            The reference solution as mesh object containing three components.
        """
        me = self.dtype_u(self.init)
        me.diff[:2] = (np.exp(t), np.exp(t))
        me.alg[0] = -np.exp(t) / (2 - t)
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
            The reference solution as mesh object containing three components.
        """

        me = self.dtype_u(self.init)
        me.diff[:2] = (np.exp(t), np.exp(t))
        me.alg[0] = (np.exp(t) * (t - 3)) / ((2 - t) ** 2)
        return me
