"""
Derived problem classes for the SDC order-reduction playground.

This module defines problem classes that extend the standard FEniCS-based 1D
heat equation implementations to demonstrate boundary lifting as a remedy for
order reduction in SDC with time-dependent Dirichlet boundary conditions.
"""

import dolfin as df
import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass_timebc


class fenics_heat_mass_timebc_lift(fenics_heat_mass_timebc):
    r"""
    One-dimensional heat equation with time-dependent Dirichlet boundary
    conditions, solved by **boundary lifting** to restore the full SDC
    convergence order.

    **Background**

    When the time-dependent BC is applied directly inside ``solve_system``
    (as in :class:`fenics_heat_mass_timebc`), the BC imposition overwrites
    rows of the right-hand-side at each SDC sweep, which means the fixed
    point of the implicit sweeper no longer matches the collocation solution.
    This causes *order reduction*: the effective convergence order is lower
    than the theoretical SDC order :math:`2M-1`.

    **Boundary lifting** eliminates this problem by reformulating the
    equation in terms of a new variable :math:`v = u - E`, where :math:`E`
    is a *lift function* that already satisfies the time-dependent BCs at
    every time. The equation for :math:`v` then has **homogeneous** Dirichlet
    BCs, and the standard SDC sweep applies without any BC imposition inside
    ``solve_system``. This restores the full collocation order.

    **Problem formulation**

    The original problem is

    .. math::
        u_t = \nu u_{xx} + f(x,t), \quad u(0,t) = \cos(0)\cos(t) = \cos(t),
        \quad u(1,t) = \cos(\pi)\cos(t) = -\cos(t),

    with exact solution :math:`u(x,t) = \cos(\pi x)\cos(t) + c`.

    We choose the lift

    .. math::
        E(x,t) = (1 - 2x)\cos(t) + c,

    which interpolates linearly between the two boundary values
    :math:`E(0,t) = \cos(t) + c` and :math:`E(1,t) = -\cos(t) + c`.

    The transformed variable :math:`v = u - E` satisfies :math:`v = 0` on
    :math:`\partial\Omega` and

    .. math::
        v_t = \nu v_{xx} + \tilde{f}(x,t),

    where the modified forcing is

    .. math::
        \tilde{f}(x,t) = f(x,t) - E_t(x,t) + \nu E_{xx}(x,t).

    Since :math:`E` is linear in :math:`x`, we have :math:`E_{xx} = 0` and
    :math:`E_t = -(1-2x)\sin(t)`, so

    .. math::
        \tilde{f}(x,t) = -\cos(\pi x)(\sin(t) - \nu\pi^2\cos(t))
                         + (1-2x)\sin(t).

    The exact solution of the transformed problem is

    .. math::
        v(x,t) = \cos(\pi x)\cos(t) - (1-2x)\cos(t) = \cos(t)(\cos(\pi x) - 1 + 2x).

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution (number of degrees of freedom). Default ``128``.
    t0 : float, optional
        Starting time. Default ``0.0``.
    family : str, optional
        FEniCS finite element family. Default ``'CG'``.
    order : int, optional
        Finite element polynomial order. Default ``4``.
    refinements : int, optional
        Number of mesh refinements. Default ``1``.
    nu : float, optional
        Diffusion coefficient :math:`\nu`. Default ``0.1``.
    c : float, optional
        Constant offset in the boundary data. Default ``0.0``.

    Attributes
    ----------
    V : FunctionSpace
        FEniCS function space.
    M : Matrix
        Mass matrix :math:`\int_\Omega u v\,dx`.
    K : Matrix
        Stiffness matrix :math:`-\nu\int_\Omega \nabla u \cdot \nabla v\,dx`.
    g : Expression
        Modified forcing term :math:`\tilde{f}` including the lift correction.
    bc_hom : DirichletBC
        Homogeneous Dirichlet BC for the transformed variable :math:`v`.

    References
    ----------
    .. [1] Spectral Deferred Correction Methods for Ordinary Differential Equations.
        A. Dutt, L. Greengard, V. Rokhlin. Mathematics of Computation, 2001.
        https://dl.acm.org/doi/abs/10.1090/S0025-5718-01-01362-X
    """

    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, c=0.0):
        """Initialization routine"""

        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

        # Override the forcing term to include lift correction terms.
        # Lift: E(x, t) = (1 - 2*x) * cos(t) + c  (linear interpolation of boundary data)
        #   dE/dt    = -(1 - 2*x) * sin(t)
        #   E_xx     = 0  (E is linear in x)
        # Modified forcing: f_tilde = f - dE/dt + nu * E_xx
        #   = -cos(pi*x) * (sin(t) - nu*pi^2*cos(t)) + (1 - 2*x) * sin(t)
        self.g = df.Expression(
            '-cos(a*x[0]) * (sin(t) - b*a*a*cos(t)) + (1 - 2*x[0]) * sin(t)',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for the transformed problem :math:`(M - \text{factor} \cdot A)\,v = \text{rhs}`.

        Uses homogeneous Dirichlet BCs since the transformed variable
        :math:`v = u - E` satisfies :math:`v = 0` on :math:`\partial\Omega`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess (not used here).
        t : float
            Current time.

        Returns
        -------
        u : dtype_u
            Solution of the transformed variable :math:`v`.
        """

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.bc_hom.apply(T, b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution of the transformed variable :math:`v` at time :math:`t`.

        The exact transformed solution is

        .. math::
            v(x,t) = u_{\text{exact}}(x,t) - E(x,t) = \cos(t)(\cos(\pi x) - 1 + 2x),

        where :math:`E(x,t) = (1-2x)\cos(t) + c` is the lift.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution of the transformed variable :math:`v`.
        """

        # v_exact = u_real_exact - E
        #         = (cos(pi*x)*cos(t) + c) - ((1 - 2*x)*cos(t) + c)
        #         = cos(t) * (cos(pi*x) - 1 + 2*x)
        u0 = df.Expression('cos(t) * (cos(a*x[0]) - 1 + 2*x[0])', a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me
