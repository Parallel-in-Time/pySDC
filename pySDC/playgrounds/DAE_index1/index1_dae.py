r"""
Index-1 semi-explicit DAE problem class
========================================

Implements the index-1 semi-explicit DAE

.. math::
    y'(t) = -\lambda y(t) + z(t) + (\lambda - 1)\sin(t),

.. math::
    0 = z(t) - (y(t) + \cos(t)),

which has the analytical solution

.. math::
    y_{\mathrm{ex}}(t) = \sin(t), \quad z_{\mathrm{ex}}(t) = \sin(t) + \cos(t).

The system is index-1 because the algebraic constraint uniquely determines
:math:`z` as a function of :math:`y` and :math:`t`.  Differentiating the
constraint once yields :math:`z' = y' - \sin(t)`, which can be substituted
back to eliminate :math:`z` and produce a pure ODE.
"""

import numpy as np

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class index1_semiexplicit_dae(ProblemDAE):
    r"""
    Index-1 semi-explicit DAE with known analytical solution.

    The system is

    .. math::
        y'(t) = -\lambda y(t) + z(t) + (\lambda - 1)\sin(t),

    .. math::
        0 = z(t) - (y(t) + \cos(t)).

    The exact solution is

    .. math::
        y_{\mathrm{ex}}(t) = \sin(t), \quad z_{\mathrm{ex}}(t) = \sin(t) + \cos(t).

    Parameters
    ----------
    lam : float, optional
        Stiffness parameter :math:`\lambda > 0`. Default 1.
    newton_tol : float, optional
        Tolerance for the nonlinear solver. Default 1e-12.
    """

    def __init__(self, lam=1.0, newton_tol=1e-12):
        """Initialization routine."""
        # 1 differential variable (y) + 1 algebraic variable (z)
        super().__init__(nvars=1, newton_tol=newton_tol)
        self._makeAttributeAndRegister('lam', localVars=locals(), readOnly=True)

    def eval_f(self, u, du, t):
        r"""
        Evaluate the implicit residual :math:`F(u, u', t)` of the semi-explicit DAE.

        For the SemiImplicitDAE sweeper the unknowns at each collocation node
        are :math:`(y', z)`.  The residual is

        .. math::
            F_{\mathrm{diff}}  &= y'(t) - \bigl(-\lambda y + z + (\lambda-1)\sin t\bigr) = 0, \\
            F_{\mathrm{alg}}   &= z - (y + \cos t) = 0.

        Parameters
        ----------
        u : dtype_u
            Current solution; ``u.diff[0]`` = :math:`y`, ``u.alg[0]`` = :math:`z`.
        du : dtype_u
            Derivative estimate; ``du.diff[0]`` = :math:`y'`, ``du.alg[0]`` = :math:`z`.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Residual with components ``.diff`` and ``.alg``.
        """
        f = self.dtype_f(self.init)
        lam = self.lam

        # Differential equation residual: y' - f(y, z, t) = 0
        f.diff[0] = du.diff[0] - (-lam * u.diff[0] + u.alg[0] + (lam - 1) * np.sin(t))
        # Algebraic constraint residual: z - (y + cos t) = 0
        f.alg[0] = u.alg[0] - (u.diff[0] + np.cos(t))

        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        r"""
        Exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        me : dtype_u
            Exact solution: ``me.diff[0]`` = :math:`\sin t`,
            ``me.alg[0]`` = :math:`\sin t + \cos t`.
        """
        me = self.dtype_u(self.init)
        me.diff[0] = np.sin(t)
        me.alg[0] = np.sin(t) + np.cos(t)
        return me

    def du_exact(self, t):
        r"""
        Exact derivative of the solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        me : dtype_u
            Exact derivative: ``me.diff[0]`` = :math:`\cos t`,
            ``me.alg[0]`` = :math:`\cos t - \sin t`.
        """
        me = self.dtype_u(self.init)
        me.diff[0] = np.cos(t)
        me.alg[0] = np.cos(t) - np.sin(t)
        return me
