import firedrake as fd

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import (
    OUTFLOW,
    cylinder_mesh,
    drag_lift,
    velocity_bcs,
)


class NavierStokes2DProjectionFiredrake(FormFiredrakeIMEX):
    r"""
    Incompressible Navier-Stokes flow around a cylinder (the DFG benchmark 2D-3) with a projection method, IMEX, with
    Firedrake, with the standard
    :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass` sweeper and the
    convergence controller :class:`~pySDC.projects.StroemungsRaum.convergence_controllers.prepare_step.PrepareStep`,
    which tells it where each step starts.

    The problem is posed for the velocity :math:`u` alone, with the viscous term implicit and the convection explicit,

    .. math::
        \int u_t \cdot v\,dx = -\nu \int \nabla u : \nabla v\,dx - \int (u \cdot \nabla u) \cdot v\,dx,

    which is what ``FormFiredrakeIMEX`` gets as forms. Incompressibility enters through the solves: after the implicit
    solve for a tentative velocity :math:`u^*` (with the no-slip and inflow conditions at the node's time), the sweeper
    passes the step :math:`\Delta\tau` from the start of the step to the node, and

    .. math::
        \int \nabla p \cdot \nabla q\,dx = -\frac{1}{\Delta\tau} \int (\nabla \cdot u^*)\, q\,dx, \quad
        \int u \cdot v\,dx = \int u^* \cdot v\,dx - \Delta\tau \int \nabla p \cdot v\,dx

    project it onto (approximately) divergence-free fields, with :math:`p = 0` at the outflow and the velocity
    conditions again in the second solve. The pressure is kept in ``pn``.

    Parameters
    ----------
    nu : float, optional
        Kinematic viscosity :math:`\nu`.
    order : int, optional
        Degree of the velocity elements; the pressure is one degree lower.
    mesh : firedrake.MeshGeometry, optional
        The mesh, by default :func:`cylinder_mesh`.
    LHS_cache_size : int, optional
        Number of solvers for the tentative velocity kept, see ``FormFiredrake``.
    """

    def __init__(self, nu=0.001, order=2, mesh=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        mesh = cylinder_mesh() if mesh is None else mesh
        V = fd.VectorFunctionSpace(mesh, 'CG', order)
        Q = fd.FunctionSpace(mesh, 'CG', order - 1)

        def mass(u, v):
            return fd.inner(u, v) * fd.dx

        def viscous(u, v):
            return -nu * fd.inner(fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx

        def convection(u, v):
            return -fd.inner(fd.dot(u, fd.nabla_grad(u)), v) * fd.dx

        t = fd.Constant(0.0)
        super().__init__(
            V,
            mass,
            viscous,
            convection,
            bcs=velocity_bcs(V, t),
            t=t,
            u_exact=fd.as_vector([0.0, 0.0]),
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('nu', 'order', localVars=locals(), readOnly=True)
        self.mesh = mesh

        # pressure Poisson equation and velocity correction, set up once: the step and u* come in through buffers
        self.pn = fd.Function(Q)
        #: the start of the current step, set by :meth:`prepare_step`
        self._t0 = None
        self._dtau = fd.Constant(1.0)
        self._u_star = fd.Function(V)
        self._u_new = fd.Function(V)
        p, q = fd.TrialFunction(Q), fd.TestFunction(Q)
        poisson = fd.LinearVariationalProblem(
            fd.inner(fd.grad(p), fd.grad(q)) * fd.dx,
            -(1 / self._dtau) * fd.div(self._u_star) * q * fd.dx,
            self.pn,
            bcs=fd.DirichletBC(Q, 0.0, OUTFLOW),
            constant_jacobian=True,
        )
        u, v = fd.TrialFunction(V), fd.TestFunction(V)
        correction = fd.LinearVariationalProblem(
            fd.inner(u, v) * fd.dx,
            fd.inner(self._u_star, v) * fd.dx - self._dtau * fd.inner(fd.grad(self.pn), v) * fd.dx,
            self._u_new,
            # the same boundary values as the tentative velocity: the data, or restricted values on coarse levels
            bcs=self._solve_bcs,
            constant_jacobian=True,
        )
        self._poisson = fd.LinearVariationalSolver(poisson)
        self._correction = fd.LinearVariationalSolver(correction)

    def prepare_step(self, t0, dt, coll):
        """
        Remember the start of the step: the projection uses the time from there to the node, see the class docstring.
        Called once per step by :class:`~pySDC.projects.StroemungsRaum.convergence_controllers.prepare_step.PrepareStep`.

        Args:
            t0 (float): Start of the step
            dt (float): Step size
            coll: The sweeper's collocation rule
        """
        self._t0 = t0

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve for the tentative velocity, :math:`M u^* - \text{factor}\, R_I(u^*) = \text{rhs}`, and project it with the
        step :math:`\Delta\tau = t - t_0` from the start of the step, see the class docstring. The pressure is kept in
        ``pn``.

        Args:
            rhs (firedrake_dual_mesh): Right-hand side, a load vector
            factor (float): Prefactor of the implicit right-hand side
            u0 (dtype_u): Initial guess
            t (float): Time of the node

        Returns:
            dtype_u: The velocity
        """
        if self._t0 is None:
            raise RuntimeError('The projection needs the start of the step: add the PrepareStep convergence controller')
        u = super().solve_system(rhs, factor, u0, t)
        self._u_star.assign(u.functionspace)
        self._dtau.assign(t - self._t0)
        self._poisson.solve()
        self._correction.solve()
        u.functionspace.assign(self._u_new)
        return u

    def drag_lift(self, u):
        """
        Drag and lift coefficients of the cylinder with the pressure of the last solve, see :func:`drag_lift`.

        Args:
            u (dtype_u): Velocity

        Returns:
            tuple of float: Drag and lift coefficients
        """
        return drag_lift(u.functionspace, self.pn, self.nu, self.mesh)
