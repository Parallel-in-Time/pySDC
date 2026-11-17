import firedrake as fd

from pySDC.helpers.ufl_forms import time_derivative
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import navier_stokes_forms


class NavierStokes2DTaylorGreenFiredrake(FormFiredrake):
    r"""
    Incompressible Navier-Stokes with a manufactured, travelling Taylor-Green solution on :math:`[-0.5, 0.5]^2`,
    monolithic, with Firedrake and given by its weak forms. This is the port of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS.fenics_NSE_2D_TaylorGreen`.

    The exact solution

    .. math::
        u = \begin{pmatrix} 1 - e^{-8\pi^2\nu t} \sin(2\pi(x - t)) \sin(\pi y) \cos(\pi y) \\
        -e^{-8\pi^2\nu t} \cos(2\pi(x - t)) \cos^2(\pi y) \end{pmatrix}, \quad
        p = 1 + \frac{4}{17} e^{-16\pi^2\nu t} \cos(4\pi(x - t)) \cos(\pi y)

    is divergence free, one-periodic in :math:`x` and constant on :math:`y = \pm 0.5`. So the same solution can be
    computed with time-dependent Dirichlet values on :math:`x = \pm 0.5` or with periodic boundaries there, which
    isolates the order reduction caused by the time-dependent boundary data (see the project's README).

    The forcing :math:`g = u_t + u \cdot \nabla u - \nu \Delta u + \nabla p` is derived from the exact solution by UFL
    and enters the form as an expression. The forms are those of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake.NavierStokes2DMonolithicFiredrake`
    plus the forcing, and everything else comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`.

    Parameters
    ----------
    nelems : int, optional
        Number of squares per direction, each split into two triangles.
    order : int, optional
        Degree of the velocity elements (Taylor-Hood).
    nu : float, optional
        Kinematic viscosity :math:`\nu`.
    periodic : bool, optional
        Periodic instead of time-dependent Dirichlet boundaries on :math:`x = \pm 0.5`.
    differentiated_bc : bool, optional
        Impose the time-dependent boundary data in differentiated form, see :meth:`prepare_step`. Needs time-dependent
        data (not `periodic`) and the sweeper
        :class:`~pySDC.projects.StroemungsRaum.sweepers.generic_implicit_mass_diffbc.generic_implicit_mass_diffbc`.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, by default Newton with tight tolerances.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(
        self,
        nelems=32,
        order=2,
        nu=0.02,
        periodic=False,
        differentiated_bc=False,
        solver_parameters=None,
        LHS_cache_size=12,
    ):
        """Initialization, see the class docstring for the arguments."""
        # boundary ids: 1, 2 at x = -+0.5, 3, 4 at y = -+0.5
        if periodic:
            mesh = fd.PeriodicRectangleMesh(nelems, nelems, 1, 1, direction='x')
            mesh.coordinates.dat.data[:] -= 0.5
        else:
            mesh = fd.RectangleMesh(nelems, nelems, 0.5, 0.5, originX=-0.5, originY=-0.5)
        V = fd.VectorFunctionSpace(mesh, 'CG', order)
        W = V * fd.FunctionSpace(mesh, 'CG', order - 1)

        t = fd.Constant(0.0)
        x, y = fd.SpatialCoordinate(mesh)
        decay = fd.exp(-8 * fd.pi**2 * nu * t)
        u_ex = fd.as_vector(
            [
                1 - decay * fd.sin(2 * fd.pi * (x - t)) * fd.sin(fd.pi * y) * fd.cos(fd.pi * y),
                -decay * fd.cos(2 * fd.pi * (x - t)) * fd.cos(fd.pi * y) ** 2,
            ]
        )
        p_ex = 1 + 4 / 17 * decay**2 * fd.cos(4 * fd.pi * (x - t)) * fd.cos(fd.pi * y)

        forcing = (
            time_derivative(u_ex, t) + fd.dot(u_ex, fd.nabla_grad(u_ex)) - nu * fd.div(fd.grad(u_ex)) + fd.grad(p_ex)
        )

        mass, rhs = navier_stokes_forms(nu, forcing)

        # the exact solution is constant on y = +-0.5
        bcs = (
            fd.DirichletBC(W.sub(0), fd.Constant((1.0, 0.0)), (3, 4)),
            fd.DirichletBC(W.sub(1), fd.Constant(1.0), (3, 4)),
        )
        if not periodic:
            bcs += (fd.DirichletBC(W.sub(0), u_ex, (1, 2)), fd.DirichletBC(W.sub(1), p_ex, (1, 2)))
        if differentiated_bc and periodic:
            raise ValueError('differentiated_bc has no effect without time-dependent boundary data')
        if solver_parameters is None:
            solver_parameters = {'snes_atol': 1e-10, 'snes_rtol': 1e-9}

        super().__init__(
            W,
            mass,
            rhs,
            bcs=bcs,
            t=t,
            u_exact=fd.as_vector([u_ex[0], u_ex[1], p_ex]),
            solver_parameters=solver_parameters,
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister(
            'nelems', 'order', 'nu', 'periodic', 'differentiated_bc', localVars=locals(), readOnly=True
        )
        self.mesh = mesh
        #: the time derivative of the exact solution, whose boundary values the differentiated conditions integrate
        self._rate = time_derivative(self.u_exact_expr, t)

    def prepare_step(self, t0, dt, coll):
        r"""
        Impose the boundary data of the step's collocation nodes in differentiated form. Rather than evaluating the
        data at the node, :math:`g(\tau_m)`, the boundary values are recovered from its time derivative by the
        collocation quadrature,

        .. math::
            g_m = g(t_0) + \Delta t \sum_j Q_{mj}\, \dot{g}(\tau_j),

        which is consistent with the collocation polynomial instead of exact pointwise; the two differ by the
        quadrature error. That recovers most of the order the time-dependent data costs, see the project's README.
        The derivative comes from UFL. The values are handed to the solves as the boundary values to impose at the
        node times, the mechanism coarse levels use for the restricted fine solution
        (:meth:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake.use_restricted_boundary_values`),
        so on coarse levels of MLSDC and PFASST the restriction takes over, as it should.

        Called once per step by
        :class:`~pySDC.projects.StroemungsRaum.sweepers.generic_implicit_mass_diffbc.generic_implicit_mass_diffbc`.

        Args:
            t0 (float): Start of the step
            dt (float): Step size
            coll: The sweeper's collocation rule
        """
        if not self.differentiated_bc:
            raise RuntimeError(
                'prepare_step imposes differentiated boundary conditions, which this problem was not set up for; use '
                'generic_implicit_mass or pass differentiated_bc=True'
            )
        times = [t0 + dt * tau for tau in coll.nodes]
        rates = []
        for tau in times:
            self.t.assign(tau)
            rates.append(self.dtype_u(self.u_init.functionspace.interpolate(self._rate)))
        values = []
        for m in range(coll.num_nodes):
            value = self.u_exact(t0)
            for j, rate in enumerate(rates):
                value += dt * coll.Qmat[m + 1, j + 1] * rate
            values.append(value)
        self.use_restricted_boundary_values(values, times)
