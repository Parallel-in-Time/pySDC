from functools import cached_property

import firedrake as fd
from mpi4py import MPI

from pySDC.helpers.ufl_forms import time_derivative
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX


class Heat1DForcedFiredrake(FormFiredrakeIMEX):
    r"""
    Forced 1D heat equation with Dirichlet boundary values and Firedrake finite elements, IMEX with the diffusion
    implicit, in the mass-matrix formulation and given by its weak forms.

    The equation

    .. math::
        \frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2} + f

    on :math:`[0, 1]` has the exact solution

    .. math::
        u(x, t) = \sin(\pi x) \cos(t) + c, \quad \text{or} \quad u(x, t) = \cos(\pi x) \cos(t) + c

    with `time_dependent_bc`, whose Dirichlet values are then :math:`\pm \cos(t) + c`. The forcing
    :math:`f = u_t - \nu u_{xx}` is derived from the exact solution by UFL and enters the form as an expression. The
    weak formulation

    .. math::
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx

    gives :math:`M u' = -\nu K u + F`, with the diffusion implicit and the forcing explicit. The mass matrix is never
    inverted: use the mass-matrix sweepers, such as
    :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`. Everything else comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrakeIMEX`.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c : float, optional
        Constant shift :math:`c` of the solution.
    order : int, optional
        Degree of the Lagrange elements.
    time_dependent_bc : bool, optional
        Use the solution with time-dependent boundary values.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`, for instance from a ``MeshHierarchy`` for several levels.
    comm : MPI communicator, optional
        Communicator for spatial parallelism, ignored if `mesh` is given.
    """

    def __init__(self, n=30, nu=0.1, c=0.0, order=4, time_dependent_bc=False, LHS_cache_size=12, mesh=None, comm=None):
        """Initialization, see the class docstring for the arguments."""
        comm = MPI.COMM_WORLD if comm is None else comm
        self.mesh = fd.UnitIntervalMesh(n, comm=comm) if mesh is None else mesh
        V = fd.FunctionSpace(self.mesh, 'CG', order)

        t = fd.Constant(0.0)
        x = fd.SpatialCoordinate(self.mesh)[0]
        shape = fd.cos(fd.pi * x) if time_dependent_bc else fd.sin(fd.pi * x)
        u_exact = shape * fd.cos(t) + c
        forcing = time_derivative(u_exact, t) - nu * fd.div(fd.grad(u_exact))

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
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister(
            'n', 'order', 'nu', 'c', 'time_dependent_bc', 'comm', localVars=locals(), readOnly=True
        )

    @cached_property
    def x(self):
        """
        Spatial coordinate of the mesh, cached after the first access.

        Returns
        -------
        x : ufl.SpatialCoordinate
            Symbolic spatial coordinate, used to interpolate expressions on the mesh.
        """
        return fd.SpatialCoordinate(self.mesh)
