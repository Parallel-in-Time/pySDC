import logging

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh


class fenics_burgers_mass(Problem):
    r"""
    Viscous Burgers equation in one dimension, in mass-matrix form.

    .. math::
        \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}
        = \nu \frac{\partial^2 u}{\partial x^2}

    on :math:`[0, 1]` with natural (homogeneous Neumann) boundary conditions, discretised with
    ``FEniCS``. In weak form, with :math:`M` the mass matrix,

    .. math::
        M \vec{u}' = F(\vec{u}), \qquad
        F(\vec{u}) = -\left(u u_x, v\right) - \nu \left(u_x, v_x\right).

    ``eval_f`` returns the assembled :math:`F`, a load vector, and ``solve_system`` takes a
    right-hand side that is already in the dual space. The mass matrix is never inverted.

    The nonlinearity is quadratic *advection* rather than a reaction term, which is what makes this
    example worth having next to Gray-Scott: advection is where the order of the prolongation stops
    being a tuning knob and becomes a stability requirement.

    Parameters
    ----------
    c_nvars : int, optional
        Number of cells on the base mesh.
    family : str, optional
        Finite element family.
    order : int, optional
        Element order. High order is the point here: the solution is smooth for moderate ``nu``, and
        mass lumping is not available at this order anyway.
    refinements : int, optional
        Number of mesh refinements, one entry per level for a multilevel run.
    nu : float, optional
        Viscosity. Keep it moderate: as it shrinks the solution develops a steep front, high-order
        CG starts to oscillate without stabilisation, and the coarse level stops being able to carry
        the solution.
    newton_tol, newton_rtol : float, optional
        Absolute and relative tolerance of the node-local Newton solve. Both have to be tighter than
        the SDC residual tolerance or the node-local solve sets the accuracy floor.
    newton_maxiter : int, optional
        Maximum number of node-local Newton iterations.

    Attributes
    ----------
    V : FunctionSpace
        Function space of the problem.
    M : scalar
        Mass matrix.
    w : Function
        Working function carrying the current Newton iterate; ``F`` is written in terms of it.
    F : Form
        Weak form of the right-hand side.
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh

    def __init__(
        self,
        c_nvars=128,
        family='CG',
        order=4,
        refinements=0,
        nu=0.02,
        newton_tol=1e-12,
        newton_rtol=1e-11,
        newton_maxiter=100,
    ):
        """Initialization routine"""

        warning_level = getattr(df, 'WARNING', None)
        if warning_level is None and hasattr(df, 'LogLevel'):
            warning_level = df.LogLevel.WARNING
        if warning_level is not None:
            df.set_log_level(warning_level)
        logging.getLogger('FFC').setLevel(logging.WARNING)

        df.parameters['form_compiler']['optimize'] = True
        df.parameters['form_compiler']['cpp_optimize'] = True

        mesh = df.UnitIntervalMesh(c_nvars)
        num_refinements = refinements if isinstance(refinements, int) else sum(refinements)
        for _ in range(num_refinements):
            mesh = df.refine(mesh)

        self.V = df.FunctionSpace(mesh, family, order)

        super().__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 'family', 'order', 'refinements', 'nu', localVars=locals(), readOnly=True
        )
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_rtol', 'newton_maxiter', localVars=locals(), readOnly=False
        )

        self.w = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.F = (-self.w * self.w.dx(0) * v - self.nu * self.w.dx(0) * v.dx(0)) * df.dx

        u, q = df.TrialFunction(self.V), df.TestFunction(self.V)
        self.M = df.assemble(u * q * df.dx)

    def apply_mass_matrix(self, u):
        r"""
        Apply the mass matrix, :math:`M \vec{u}`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M \vec{u}`.
        """

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me

    def eval_f(self, u, t):
        r"""
        Evaluate the right-hand side in weak (dual) form, without inverting the mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The assembled load vector :math:`F(\vec{u})`.
        """

        f = self.dtype_f(self.V)
        self.w.assign(u.values)
        f.values = df.Function(self.V, df.assemble(self.F))

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`M \vec{u} - factor \cdot F(\vec{u}) = \vec{rhs}` by Newton, rhs already dual.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side, in the dual space.
        factor : float
            Node-to-node stepsize.
        u0 : dtype_u
            Initial guess for the Newton iteration.
        t : float
            Current time.

        Returns
        -------
        sol : dtype_u
            Solution.
        """

        sol = self.dtype_u(u0)
        self.w.assign(sol.values)

        du = df.TrialFunction(self.V)
        Jform = df.derivative(self.F, self.w, du)

        res, delta, res0 = df.Function(self.V), df.Function(self.V), None

        for _ in range(self.newton_maxiter):
            self.M.mult(self.w.vector(), res.vector())
            res.vector().axpy(-factor, df.assemble(self.F))
            res.vector().axpy(-1.0, rhs.values.vector())

            norm = res.vector().norm('l2')
            res0 = norm if res0 is None else res0
            if norm < self.newton_tol or norm < self.newton_rtol * res0:
                break

            df.solve(self.M - factor * df.assemble(Jform), delta.vector(), res.vector())
            self.w.vector().axpy(-1.0, delta.vector())

        sol.values.assign(self.w)

        return sol

    def u_exact(self, t):
        r"""
        Initial condition :math:`u(x, 0) = \cos(2 \pi x)`, only available at :math:`t = 0`.

        It has vanishing derivative at both ends, so it is consistent with the natural boundary
        conditions, and it is smooth, so the coarse levels can carry it.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Initial condition.
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        expr = df.Expression('cos(2*pi*x[0])', degree=max(1, self.order), pi=np.pi)

        return self.dtype_u(df.interpolate(expr, self.V))
