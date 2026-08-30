"""FEniCS problem classes for the SDC order-reduction playground."""

import dolfin as df
import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass_timebc


class fenics_heat_mass_timebc_lift(fenics_heat_mass_timebc):
    r"""Heat equation with time-dependent BCs solved after boundary lifting.

    The physical problem has the exact solution

    .. math::
        u(x,t) = \cos(\pi x)\cos(t) + c.

    The controller integrates the shifted variable

    .. math::
        v = u - E, \qquad E(x,t) = (1-2x)\cos(t)+c,

    for which the boundary conditions are homogeneous. Since

    .. math::
        E_t = -(1-2x)\sin(t), \qquad E_{xx}=0,

    the shifted equation is

    .. math::
        v_t = \nu v_{xx} + \widetilde f,
        \qquad \widetilde f = f - E_t + \nu E_{xx}
        = f + (1-2x)\sin(t).

    This class retains the historical playground API: ``u_exact`` returns the
    exact lifted solution because that is the state integrated by pySDC. The
    :class:`fenics_heat_mass_timebc_lift_physical` adapter below exposes the
    unchanged physical cosine exact solution while reusing this implementation.
    """

    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, c=0.0):
        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

        self.g = df.Expression(
            '-cos(a*x[0]) * (sin(t) - b*a*a*cos(t)) + (1 - 2*x[0]) * sin(t)',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

    def eval_f(self, u, t):
        """Evaluate the lifted IMEX right-hand side for ``v``."""
        f = self.dtype_f(self.V)
        self.K.mult(u.values.vector(), f.impl.values.vector())

        self.g.t = t
        f.expl = self.apply_mass_matrix(self.dtype_u(df.interpolate(self.g, self.V)))
        return f

    def solve_system(self, rhs, factor, u0, t):
        """Solve the lifted system with homogeneous Dirichlet conditions."""
        v = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)
        self.bc_hom.apply(T, b.values.vector())
        df.solve(T, v.values.vector(), b.values.vector())
        return v

    def fix_residual(self, res):
        """Apply homogeneous boundary conditions to a lifted residual."""
        self.bc_hom.apply(res.values.vector())
        return None

    def u_exact(self, t):
        """Return the exact lifted state ``v = u - E``."""
        v_exact = df.Expression(
            'cos(t) * (cos(a*x[0]) - 1 + 2*x[0])',
            a=np.pi,
            t=t,
            degree=self.order,
        )
        return self.dtype_u(df.interpolate(v_exact, self.V), val=self.V)


class fenics_heat_mass_timebc_lift_physical(fenics_heat_mass_timebc_lift):
    r"""Boundary-lifted problem with the physical cosine exact solution.

    The numerical state remains the lifted variable ``v``. The physical
    solution is reconstructed with ``u = v + E`` by the driver. This adapter
    therefore needs only the physical exact solution and the lift helper; all
    numerical problem methods are inherited from
    :class:`fenics_heat_mass_timebc_lift`.
    """

    def lift(self, t):
        """Return the finite-element representation of ``E(., t)``."""
        E = df.Expression('(1 - 2*x[0]) * cos(t) + c', c=self.c, t=t, degree=self.order)
        return self.dtype_u(df.interpolate(E, self.V), val=self.V)

    def u_exact_lifted(self, t):
        """Return the exact internal state ``v = u - E``."""
        return super().u_exact(t)

    def u_exact(self, t):
        r"""Return the unchanged physical exact solution.

        .. math::
            u(x,t) = \cos(\pi x)\cos(t)+c.
        """
        u_exact = df.Expression(
            'cos(a*x[0]) * cos(t) + c',
            a=np.pi,
            c=self.c,
            t=t,
            degree=self.order,
        )
        return self.dtype_u(df.interpolate(u_exact, self.V), val=self.V)
