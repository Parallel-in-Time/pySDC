"""
Problem classes for the 1D Allen-Cahn FEniCS playground.

Demonstrates SDC order reduction caused by inhomogeneous time-dependent
Dirichlet boundary conditions, and how boundary lifting can restore the
full convergence order.

Two classes are provided:

* :class:`fenics_allencahn_imex_timebc` — the Allen-Cahn equation with
  naive time-dependent BC imposition (causes order reduction).
* :class:`fenics_allencahn_imex_timebc_lift` — the same equation
  reformulated via boundary lifting so that the transformed variable
  satisfies **homogeneous** BCs (restores full SDC order).
"""

import logging

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


class fenics_allencahn_imex_timebc(Problem):
    r"""
    One-dimensional Allen-Cahn equation on :math:`[0, 1]` with inhomogeneous
    time-dependent Dirichlet boundary conditions, solved with an IMEX split
    in space (diffusion implicit, nonlinear reaction explicit) and FEniCS as
    the spatial discretisation.

    **Equation**

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}
        - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u) - 6 d_w u (1 - u),

    with exact (traveling-wave) solution

    .. math::
        u(x, t) = \tfrac{1}{2}\!\left(1
        + \tanh\!\left(\frac{x - 0.5 - v\,t}{\sqrt{2}\,\varepsilon}\right)\right),
        \qquad v = 3\sqrt{2}\,\varepsilon\,d_w.

    The spatial domain is mapped from :math:`[-0.5, 0.5]` to :math:`[0, 1]`
    by the substitution :math:`x \to x - 0.5`.

    **Boundary conditions** are inhomogeneous and time-dependent (exact
    solution evaluated at :math:`x = 0` and :math:`x = 1`).

    **Order reduction mechanism**

    In :meth:`solve_system`, the time-dependent Dirichlet BC is enforced via

    .. code-block:: python

        self.bc.apply(T, b.values.vector())   # modifies system matrix
        self.bc.apply(b.values.vector())      # overwrites boundary DOFs

    The second line *overwrites* boundary entries of the accumulated SDC
    right-hand side with the exact boundary value at the new time.  This
    disrupts the implicit sweeper's fixed-point equation and leads to
    **SDC order reduction**.

    **IMEX split**

    * Implicit part: :math:`f_{\mathrm{impl}} = K u`
      (assembled stiffness matrix applied to :math:`u`).
    * Explicit part: :math:`f_{\mathrm{expl}} = M(-N(u))`
      where :math:`N(u) = \frac{2}{\varepsilon^2} u(1-u)(1-2u) + 6 d_w u(1-u)`
      is the positive reaction function, so that the PDE reads
      :math:`u_t = u_{xx} - N(u)`.

    Parameters
    ----------
    c_nvars : int, optional
        Number of mesh intervals (spatial resolution). Default ``128``.
    t0 : float, optional
        Starting time. Default ``0.0``.
    family : str, optional
        FEM element family. Default ``'CG'``.
    order : int, optional
        FEM element polynomial order. Default ``4``.
    refinements : int, optional
        Number of mesh refinements. Default ``1``.
    eps : float, optional
        Interface parameter :math:`\varepsilon`. Default ``0.3``.
    dw : float, optional
        Driving force :math:`d_w`. Default ``-0.04``.

    Attributes
    ----------
    V : dolfin.FunctionSpace
    M : dolfin.Matrix
        Mass matrix :math:`\int_\Omega u\,v\,dx`.
    K : dolfin.Matrix
        Stiffness matrix :math:`-\int_\Omega \nabla u \cdot \nabla v\,dx`.
    bc : dolfin.DirichletBC
        Time-dependent inhomogeneous Dirichlet BC.
    bc_hom : dolfin.DirichletBC
        Homogeneous Dirichlet BC (used for residual fixing and mass inversion).
    v_speed : float
        Front propagation speed :math:`v = 3\sqrt{2}\,\varepsilon\,d_w`.
    fix_bc_for_residual : bool
        Always ``True``; instructs the sweeper to call :meth:`fix_residual`.
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(
        self,
        c_nvars=128,
        t0=0.0,
        family='CG',
        order=4,
        refinements=1,
        eps=0.3,
        dw=-0.04,
    ):
        """Initialisation routine."""

        def Boundary(x, on_boundary):
            return on_boundary

        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        mesh = df.UnitIntervalMesh(c_nvars)
        for _ in range(refinements):
            mesh = df.refine(mesh)

        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        super().__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars',
            't0',
            'family',
            'order',
            'refinements',
            'eps',
            'dw',
            localVars=locals(),
            readOnly=True,
        )

        # Front speed: v = 3*sqrt(2)*eps*dw
        self.v_speed = 3.0 * np.sqrt(2.0) * eps * dw

        # Assemble mass and stiffness matrices
        u_trial = df.TrialFunction(self.V)
        v_test = df.TestFunction(self.V)
        self.M = df.assemble(u_trial * v_test * df.dx)
        self.K = df.assemble(-1.0 * df.inner(df.nabla_grad(u_trial), df.nabla_grad(v_test)) * df.dx)

        # Time-dependent Dirichlet BC (exact traveling-wave evaluated at boundary)
        self.u_D = df.Expression(
            '0.5*(1 + tanh((x[0] - 0.5 - vsp*t) / (s2*eps)))',
            vsp=self.v_speed,
            t=t0,
            s2=np.sqrt(2.0),
            eps=eps,
            degree=order,
        )
        self.bc = df.DirichletBC(self.V, self.u_D, Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), Boundary)

        self.fix_bc_for_residual = True

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve the linear system :math:`(M - \text{factor}\,K)\,u = \text{rhs}`.

        The time-dependent BC is applied via two ``bc.apply`` calls:

        * ``bc.apply(T, b.values.vector())`` — modifies the system matrix;
        * ``bc.apply(b.values.vector())`` — **overwrites** boundary DOFs of
          the right-hand side with :math:`u_D(t)`.

        The second call is the source of **SDC order reduction**: it replaces
        whatever value the SDC iteration accumulated at the boundary with the
        exact Dirichlet data at the *new* time, breaking the fixed-point
        property of the sweep.

        Parameters
        ----------
        rhs : fenics_mesh
        factor : float
        u0 : fenics_mesh
        t : float

        Returns
        -------
        fenics_mesh
        """
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        # Update expression time and impose time-dependent BC (causes order reduction)
        self.u_D.t = t
        self.bc.apply(T, b.values.vector())
        self.bc.apply(b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

    def eval_f(self, u, t):
        r"""
        Evaluate both parts of the right-hand side.

        * ``f.impl = K u`` — stiffness matrix applied to :math:`u`.
        * ``f.expl = M(-N(u))`` — mass-matrix-weighted negated reaction, where
          :math:`N(u) = \tfrac{2}{\varepsilon^2}u(1-u)(1-2u) + 6d_w u(1-u)`
          so that the Allen-Cahn PDE reads :math:`u_t = u_{xx} - N(u)`.

        Parameters
        ----------
        u : fenics_mesh
        t : float

        Returns
        -------
        rhs_fenics_mesh
        """
        f = self.dtype_f(self.V)

        # Implicit part: stiffness applied to u
        self.K.mult(u.values.vector(), f.impl.values.vector())

        # Explicit part: -N(u), mass-weighted
        # N(u) = (2/eps^2)*u*(1-u)*(1-2u) + 6*dw*u*(1-u)  [positive reaction]
        u_vec = u.values.vector()[:]
        eps2 = self.eps**2
        pos_N = (2.0 / eps2) * u_vec * (1.0 - u_vec) * (1.0 - 2.0 * u_vec) + 6.0 * self.dw * u_vec * (1.0 - u_vec)
        expl_fn = self.dtype_u(self.V)
        expl_fn.values.vector()[:] = -pos_N
        f.expl = self.apply_mass_matrix(expl_fn)

        return f

    def apply_mass_matrix(self, u):
        r"""Apply the mass matrix :math:`M` to ``u``."""
        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())
        return me

    def fix_residual(self, res):
        """Zero out boundary DOFs of the residual (required by the sweeper)."""
        self.bc_hom.apply(res.values.vector())
        return None

    def u_exact(self, t):
        r"""
        Return the exact traveling-wave solution at time :math:`t`.

        Parameters
        ----------
        t : float

        Returns
        -------
        fenics_mesh
        """
        u0_expr = df.Expression(
            '0.5*(1 + tanh((x[0] - 0.5 - vsp*t) / (s2*eps)))',
            vsp=self.v_speed,
            t=t,
            s2=np.sqrt(2.0),
            eps=self.eps,
            degree=self.order,
        )
        return self.dtype_u(df.interpolate(u0_expr, self.V), val=self.V)


class fenics_allencahn_imex_timebc_lift(fenics_allencahn_imex_timebc):
    r"""
    Same Allen-Cahn equation using **boundary lifting** to restore the full
    SDC convergence order.

    **Idea** — decompose :math:`u = v + E` where the *lift*

    .. math::
        E(x, t) = u_L(t)\,(1 - x) + u_R(t)\,x

    interpolates linearly between the two time-dependent boundary values

    .. math::
        u_L(t) = \tfrac{1}{2}\!\left(1
          + \tanh\!\left(\frac{-0.5 - v\,t}{\sqrt{2}\,\varepsilon}\right)\right),
        \qquad
        u_R(t) = \tfrac{1}{2}\!\left(1
          + \tanh\!\left(\frac{0.5 - v\,t}{\sqrt{2}\,\varepsilon}\right)\right).

    The transformed variable :math:`v = u - E` satisfies **homogeneous**
    Dirichlet BCs :math:`v(0, t) = v(1, t) = 0` at all times.  Because
    :meth:`solve_system` only applies ``bc_hom`` (no overwrite of non-zero
    values), the SDC fixed-point property is preserved and the full
    collocation order is recovered.

    **Transformed equation**

    Since the lift is linear in :math:`x` (:math:`E_{xx} = 0`), the equation
    for :math:`v` reads

    .. math::
        v_t = v_{xx} - N(v + E) - E_t,

    where :math:`E_t(x, t) = u_L'(t)(1 - x) + u_R'(t)\,x`.

    **Modified IMEX split**

    * Implicit part: :math:`f_{\mathrm{impl}} = K v` (unchanged).
    * Explicit part: :math:`f_{\mathrm{expl}} = M(-N(v + E) - E_t)`.

    Parameters
    ----------
    Same as :class:`fenics_allencahn_imex_timebc`.
    """

    def __init__(
        self,
        c_nvars=128,
        t0=0.0,
        family='CG',
        order=4,
        refinements=1,
        eps=0.3,
        dw=-0.04,
    ):
        """Initialisation routine."""
        super().__init__(c_nvars, t0, family, order, refinements, eps, dw)

    # ------------------------------------------------------------------
    # helpers for the lift and its time derivative
    # ------------------------------------------------------------------

    def _get_lift_values(self, t):
        r"""
        Nodal values of :math:`E(x, t)` at the DOF coordinates.

        Returns
        -------
        numpy.ndarray  shape ``(n_dofs,)``
        """
        vsp, eps, s2 = self.v_speed, self.eps, np.sqrt(2.0)
        uL = 0.5 * (1.0 + np.tanh((-0.5 - vsp * t) / (s2 * eps)))
        uR = 0.5 * (1.0 + np.tanh((0.5 - vsp * t) / (s2 * eps)))
        x = self.V.tabulate_dof_coordinates()[:, 0]
        return uL * (1.0 - x) + uR * x

    def _get_lift_dt_values(self, t):
        r"""
        Nodal values of :math:`\partial_t E(x, t)` at the DOF coordinates.

        Returns
        -------
        numpy.ndarray  shape ``(n_dofs,)``
        """
        vsp, eps, s2 = self.v_speed, self.eps, np.sqrt(2.0)
        sL = (-0.5 - vsp * t) / (s2 * eps)
        sR = (0.5 - vsp * t) / (s2 * eps)
        duL_dt = -vsp / (2.0 * s2 * eps * np.cosh(sL) ** 2)
        duR_dt = -vsp / (2.0 * s2 * eps * np.cosh(sR) ** 2)
        x = self.V.tabulate_dof_coordinates()[:, 0]
        return duL_dt * (1.0 - x) + duR_dt * x

    # ------------------------------------------------------------------
    # overridden problem interface
    # ------------------------------------------------------------------

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(M - \text{factor}\,K)\,v = \text{rhs}` with
        **homogeneous** Dirichlet BCs.

        Only ``bc_hom.apply(T, b.values.vector())`` is called — no boundary
        DOFs are overwritten with non-zero values, so the SDC fixed-point
        property for :math:`v` is unaffected.

        Parameters
        ----------
        rhs : fenics_mesh
        factor : float
        u0 : fenics_mesh
        t : float

        Returns
        -------
        fenics_mesh
        """
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.bc_hom.apply(T, b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

    def eval_f(self, v, t):
        r"""
        Evaluate both parts of the right-hand side for the lifted variable.

        * ``f.impl = K v`` — stiffness applied to :math:`v`.
        * ``f.expl = M(-N(v + E) - E_t)`` — mass-weighted negated reaction
          :math:`N(v+E)` plus negated lift derivative :math:`E_t`, where
          :math:`N(u) = \tfrac{2}{\varepsilon^2}u(1-u)(1-2u) + 6d_w u(1-u)`.

        Parameters
        ----------
        v : fenics_mesh
        t : float

        Returns
        -------
        rhs_fenics_mesh
        """
        f = self.dtype_f(self.V)

        # Implicit part: stiffness of v
        self.K.mult(v.values.vector(), f.impl.values.vector())

        # Lift and its time derivative at current time
        E_vals = self._get_lift_values(t)
        Et_vals = self._get_lift_dt_values(t)

        # u = v + E at DOF locations
        u_vals = v.values.vector()[:] + E_vals

        # N(v+E) = (2/eps^2)*u*(1-u)*(1-2u) + 6*dw*u*(1-u)  [positive reaction]
        eps2 = self.eps**2
        pos_N = (2.0 / eps2) * u_vals * (1.0 - u_vals) * (1.0 - 2.0 * u_vals) + 6.0 * self.dw * u_vals * (
            1.0 - u_vals
        )

        # Explicit forcing for v: -N(v+E) - E_t  (matches lifted equation v_t = v_xx - N(v+E) - E_t)
        expl_nodal = -pos_N - Et_vals

        expl_fn = self.dtype_u(self.V)
        expl_fn.values.vector()[:] = expl_nodal
        f.expl = self.apply_mass_matrix(expl_fn)

        return f

    def u_exact(self, t):
        r"""
        Exact solution of the transformed variable :math:`v = u - E` at
        time :math:`t`.

        Parameters
        ----------
        t : float

        Returns
        -------
        fenics_mesh
        """
        # Interpolate the exact traveling wave
        u_expr = df.Expression(
            '0.5*(1 + tanh((x[0] - 0.5 - vsp*t) / (s2*eps)))',
            vsp=self.v_speed,
            t=t,
            s2=np.sqrt(2.0),
            eps=self.eps,
            degree=self.order,
        )
        u_fn = df.interpolate(u_expr, self.V)

        # Subtract the lift: v = u - E
        v_fn = df.Function(self.V)
        v_fn.vector()[:] = u_fn.vector()[:] - self._get_lift_values(t)

        return self.dtype_u(v_fn, val=self.V)
