import firedrake as fd

from pySDC.helpers.ufl_forms import time_derivative
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX


class Heat2DFiredrake(FormFiredrakeIMEX):
    r"""
    Forced 2D heat equation on the unit square with time-dependent Dirichlet values, IMEX, with Firedrake and given by
    its weak forms. This is the port of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.HeatEquation_2D_FEniCS.fenics_heat2D_mass`, on the same
    mesh.

    The equation :math:`u_t = \nu \Delta u + f` has the exact solution
    :math:`u = \sin(\pi x) \sin(\pi y) \cos(t) + c`. The diffusion is treated implicitly, the forcing explicitly. The
    forcing :math:`f = u_t - \nu \Delta u` is derived from the exact solution by UFL and interpolated into the
    solution space, as the FEniCS version does with its ``Expression``. Everything else comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrakeIMEX`.

    Parameters
    ----------
    c_nvars : int, optional
        Number of squares per direction, each split into two triangles.
    order : int, optional
        Degree of the Lagrange elements.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c : float, optional
        Constant :math:`c` of the exact solution.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, c_nvars=64, order=2, nu=0.1, c=0.0, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        # `diagonal='right'` and equispaced nodes are where DOLFIN puts them, so both versions discretize alike
        mesh = fd.UnitSquareMesh(c_nvars, c_nvars, diagonal='right')
        V = fd.FunctionSpace(mesh, 'CG', order, variant='equispaced')

        t = fd.Constant(0.0)
        x, y = fd.SpatialCoordinate(mesh)
        u_exact = fd.sin(fd.pi * x) * fd.sin(fd.pi * y) * fd.cos(t) + c
        forcing = fd.interpolate(time_derivative(u_exact, t) - nu * fd.div(fd.grad(u_exact)), V)

        def mass(u, v):
            return u * v * fd.dx

        def diffusion(u, v):
            return -nu * fd.inner(fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx

        def source(u, v):
            return forcing * v * fd.dx

        super().__init__(
            V,
            mass,
            diffusion,
            source,
            bcs=(fd.DirichletBC(V, u_exact, 'on_boundary'),),
            t=t,
            u_exact=u_exact,
            # the implicit systems are linear
            solver_parameters={'snes_type': 'ksponly'},
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('c_nvars', 'order', 'nu', 'c', localVars=locals(), readOnly=True)
        self.mesh = mesh
