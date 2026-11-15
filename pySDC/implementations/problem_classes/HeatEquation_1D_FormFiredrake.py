import firedrake as fd

from pySDC.helpers.ufl_forms import time_derivative
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX


class Heat1DMassFiredrake(FormFiredrakeIMEX):
    r"""
    Forced 1D heat equation in the mass-matrix formulation, IMEX, with Firedrake and given by its weak forms. This is
    the port of
    :class:`~pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced.fenics_heat_mass` and, with
    `time_dependent_bc`, of
    :class:`~pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced.fenics_heat_mass_timebc`.

    The equation :math:`u_t = \nu u_{xx} + f` on :math:`[0, 1]` has the exact solution

    .. math::
        u(x, t) = \sin(\pi x) \cos(t) + c, \quad \text{or} \quad u(x, t) = \cos(\pi x) \cos(t) + c

    with `time_dependent_bc`, whose Dirichlet values are then :math:`\pm \cos(t) + c`. The diffusion is implicit and the
    forcing :math:`f = u_t - \nu u_{xx}` explicit. The forcing is derived from the exact solution by UFL and, as in the
    FEniCS version, interpolated into the finite element space before it is integrated.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    order : int, optional
        Degree of the Lagrange elements.
    variant : str, optional
        Where the Lagrange elements have their nodes. The default ``'equispaced'`` puts them where FEniCS does, so
        interpolated functions are the same; Firedrake's own default for high degrees is ``'spectral'``.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c : float, optional
        Constant shift :math:`c` of the solution.
    time_dependent_bc : bool, optional
        Use the solution with time-dependent boundary values.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`, e.g. from a ``MeshHierarchy`` for several levels.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(
        self,
        n=256,
        order=4,
        variant='equispaced',
        nu=0.1,
        c=0.0,
        time_dependent_bc=False,
        mesh=None,
        LHS_cache_size=12,
    ):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.UnitIntervalMesh(n) if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'CG', order, variant=variant)

        t = fd.Constant(0.0)
        x = fd.SpatialCoordinate(mesh)[0]
        shape = fd.cos(fd.pi * x) if time_dependent_bc else fd.sin(fd.pi * x)
        u_exact = shape * fd.cos(t) + c
        forcing = fd.interpolate(time_derivative(u_exact, t) - nu * fd.div(fd.grad(u_exact)), V)

        def mass(u, v):
            return u * v * fd.dx

        def diffusion(u, v):
            return -nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

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
        self._makeAttributeAndRegister('n', 'order', 'nu', 'c', 'time_dependent_bc', localVars=locals(), readOnly=True)
