import firedrake as fd

from pySDC.helpers.ufl_forms import time_derivative
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake


class NavierStokes2DTaylorGreenFiredrake(FormFiredrake):
    r"""
    Incompressible Navier-Stokes with a manufactured, travelling Taylor-Green solution on :math:`[-0.5, 0.5]^2`,
    monolithic, with Firedrake and given by its weak forms. This is the port of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS.fenics_NSE_2D_TaylorGreen`
    without its ``differentiated_bc`` option, on the same meshes.

    The exact solution

    .. math::
        u = \begin{pmatrix} 1 - e^{-8\pi^2\nu t} \sin(2\pi(x - t)) \sin(\pi y) \cos(\pi y) \\
        -e^{-8\pi^2\nu t} \cos(2\pi(x - t)) \cos^2(\pi y) \end{pmatrix}, \quad
        p = 1 + \frac{4}{17} e^{-16\pi^2\nu t} \cos(4\pi(x - t)) \cos(\pi y)

    is divergence free, one-periodic in :math:`x` and constant on :math:`y = \pm 0.5`. So the same solution can be
    computed with time-dependent Dirichlet values on :math:`x = \pm 0.5` or with periodic boundaries there, which
    isolates the order reduction caused by the time-dependent boundary data (see the project's README).

    The forcing :math:`g = u_t + u \cdot \nabla u - \nu \Delta u + \nabla p` is derived from the exact solution by UFL,
    and interpolated into Lagrange elements two degrees above the velocity, with equispaced nodes, as the FEniCS version
    does with its ``Expression``. The forms are those of
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
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, by default Newton with tolerances like those of the FEniCS version.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, nelems=32, order=2, nu=0.02, periodic=False, solver_parameters=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        # `diagonal='right'` splits the squares like DOLFIN does. Boundary ids: 1, 2 at x = -+0.5, 3, 4 at y = -+0.5.
        if periodic:
            mesh = fd.PeriodicRectangleMesh(nelems, nelems, 1, 1, direction='x', diagonal='right')
            mesh.coordinates.dat.data[:] -= 0.5
        else:
            mesh = fd.RectangleMesh(nelems, nelems, 0.5, 0.5, originX=-0.5, originY=-0.5, diagonal='right')
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
        # interpolated with equispaced nodes, where DOLFIN puts them: Firedrake's default for high degrees differs
        forcing = fd.interpolate(forcing, fd.VectorFunctionSpace(mesh, 'CG', order + 2, variant='equispaced'))

        def mass(w, z):
            u, _ = fd.split(w)
            v, _ = fd.split(z)
            return fd.inner(u, v) * fd.dx

        def rhs(w, z):
            u, p = fd.split(w)
            v, q = fd.split(z)
            return (
                -fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
                - nu * fd.inner(fd.nabla_grad(u), fd.nabla_grad(v))
                + p * fd.div(v)
                + fd.inner(forcing, v)
                + fd.div(u) * q
            ) * fd.dx

        # the exact solution is constant on y = +-0.5
        bcs = (
            fd.DirichletBC(W.sub(0), fd.Constant((1.0, 0.0)), (3, 4)),
            fd.DirichletBC(W.sub(1), fd.Constant(1.0), (3, 4)),
        )
        if not periodic:
            bcs += (fd.DirichletBC(W.sub(0), u_ex, (1, 2)), fd.DirichletBC(W.sub(1), p_ex, (1, 2)))
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
        self._makeAttributeAndRegister('nelems', 'order', 'nu', 'periodic', localVars=locals(), readOnly=True)
        self.mesh = mesh
