from functools import cached_property

import firedrake as fd
from mpi4py import MPI

from pySDC.implementations.problem_classes.HeatEquation_1D_FormFiredrake import Heat1DMassFiredrake


class Heat1DForcedFiredrake(Heat1DMassFiredrake):
    r"""
    Forced 1D heat equation with Dirichlet BCs and Firedrake finite elements, IMEX with diffusion implicit, in the
    mass-matrix formulation.

    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    For initial conditions with constant c

    .. math::
        u(x, 0) = \sin(\pi x) + c,

    the exact solution is given by

    .. math::
        u(x, t) = \sin(\pi x)\cos(t) + c.

    The problem is discretized with finite elements in the *weak formulation*

    .. math::
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx,

    i.e. :math:`M u' = -\nu K u + M f`, with the diffusion implicit and the (interpolated) forcing explicit. The mass
    matrix is never inverted: use the mass-matrix sweepers, such as
    :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`. Everything is inherited
    from :class:`~pySDC.implementations.problem_classes.HeatEquation_1D_FormFiredrake.Heat1DMassFiredrake`, with
    Firedrake's default placement of the Lagrange nodes and a communicator for spatial parallelism.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c: float, optional
        Constant for the Dirichlet boundary condition :math: `c`
    order : int, optional
        Degree of the Lagrange elements.
    LHS_cache_size : int, optional
        Cache size for variational problem solvers
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`, for instance from a hierarchy.
    comm : MPI communicator, optional
        Supply an MPI communicator for spatial parallelism, ignored if `mesh` is given.
    """

    def __init__(self, n=30, nu=0.1, c=0.0, order=4, LHS_cache_size=12, mesh=None, comm=None):
        """Initialization, see the class docstring for the arguments."""
        comm = MPI.COMM_WORLD if comm is None else comm
        self.mesh = fd.UnitIntervalMesh(n, comm=comm) if mesh is None else mesh
        super().__init__(n=n, order=order, variant=None, nu=nu, c=c, mesh=self.mesh, LHS_cache_size=LHS_cache_size)
        self._makeAttributeAndRegister('comm', localVars=locals(), readOnly=True)

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
