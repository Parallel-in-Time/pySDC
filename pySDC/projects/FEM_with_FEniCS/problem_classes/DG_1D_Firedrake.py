r"""
Discontinuous-Galerkin versions of the three 1D examples with Firedrake, given by their weak forms. These are the
ports of :mod:`pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_FEniCS`, on the same meshes: interior penalty for
diffusion, Nitsche for Dirichlet data, local Lax-Friedrichs for the Burgers advection.

The ``penalty`` argument must be pinned to the fine level's value scaled by :math:`h_l / h_0` in a multilevel run.
The SIPG form depends on the mesh and order it is built on, so rediscretising it per level changes the penalty -- the
dominant term -- and the coarse operator stops being :math:`P^T A_F P`. See ``setups.py`` for the pinned ladder.

All elements use equispaced nodes, where FEniCS puts them, so interpolated data are the same as there.
"""

import firedrake as fd
from ufl.domain import extract_unique_domain

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake, FormFiredrakeIMEX


def sipg(u, v, kappa, alpha):
    r"""
    Symmetric interior penalty form for :math:`-\nabla \cdot (\kappa \nabla u)`, natural boundaries.

    The exterior facets carry no terms, so this alone imposes homogeneous Neumann conditions. Add :func:`nitsche` on
    top for Dirichlet data.

    Args:
        u: Argument of the operator, may enter nonlinearly
        v: Test function
        kappa: Diffusion coefficient
        alpha (float): Penalty coefficient, entering as :math:`\alpha \kappa / h`

    Returns:
        ufl.Form: The form :math:`a(u, v)`, so a right-hand side uses ``-sipg(...)``
    """
    mesh = extract_unique_domain(v)
    n, h = fd.FacetNormal(mesh), fd.CellDiameter(mesh)
    return (
        kappa * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        - kappa * fd.inner(fd.avg(fd.grad(u)), fd.jump(v, n)) * fd.dS
        - kappa * fd.inner(fd.jump(u, n), fd.avg(fd.grad(v))) * fd.dS
        + alpha * kappa / fd.avg(h) * fd.inner(fd.jump(u, n), fd.jump(v, n)) * fd.dS
    )


def nitsche(u, v, u_D, kappa, alpha):
    r"""
    Weak (Nitsche) Dirichlet terms on the exterior boundary, to be added to :func:`sipg`.

    Args:
        u, v, kappa, alpha: As in :func:`sipg`
        u_D: Boundary data

    Returns:
        ufl.Form: The terms in `u`, added to the operator
        ufl.Form: The terms carrying the data, a load vector
    """
    mesh = extract_unique_domain(v)
    n, h = fd.FacetNormal(mesh), fd.CellDiameter(mesh)
    a = (-kappa * fd.dot(fd.grad(u), n) * v - kappa * u * fd.dot(fd.grad(v), n) + alpha * kappa / h * u * v) * fd.ds
    L = (-kappa * u_D * fd.dot(fd.grad(v), n) + alpha * kappa / h * u_D * v) * fd.ds
    return a, L


class Heat1DDGFiredrake(FormFiredrakeIMEX):
    r"""
    Forced 1D heat equation, DG in space, IMEX. The port of
    :class:`~pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_FEniCS.fenics_heat_dg_mass`.

    .. math::
        \frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2} + f, \quad
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t))

    on :math:`[0, 1]` with :math:`u = c` on the boundary, imposed by Nitsche terms, and the exact solution
    :math:`u = \sin(\pi x) \cos(t) + c`. The SIPG Laplacian with its Nitsche terms is implicit; the interpolated
    forcing and the Nitsche load are explicit, as in the FEniCS version.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    order : int, optional
        Degree of the elements.
    nu : float, optional
        Diffusion coefficient.
    c : float, optional
        Boundary value.
    sigma : float, optional
        Penalty constant, used only when `penalty` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`, see the module docstring.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, n=256, order=4, nu=0.1, c=0.0, sigma=10.0, penalty=None, mesh=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.UnitIntervalMesh(n) if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'DG', order, variant='equispaced')
        alpha = sigma * max(order, 1) ** 2 if penalty is None else penalty

        t = fd.Constant(0.0)
        x = fd.SpatialCoordinate(mesh)[0]
        forcing = fd.interpolate(-fd.sin(fd.pi * x) * (fd.sin(t) - nu * fd.pi**2 * fd.cos(t)), V)

        def mass(u, v):
            return u * v * fd.dx

        def rhs(u, v):
            return -(sipg(u, v, nu, alpha) + nitsche(u, v, c, nu, alpha)[0])

        def rhs_exp(u, v):
            return forcing * v * fd.dx + nitsche(u, v, c, nu, alpha)[1]

        super().__init__(
            V,
            mass,
            rhs,
            rhs_exp,
            t=t,
            u_exact=fd.sin(fd.pi * x) * fd.cos(t) + c,
            solver_parameters={'snes_type': 'ksponly'},
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'nu', 'c', 'sigma', 'penalty', localVars=locals(), readOnly=True)


class Burgers1DDGFiredrake(FormFiredrake):
    r"""
    Viscous Burgers in 1D, DG in space. The port of
    :class:`~pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_FEniCS.fenics_burgers_dg_mass`.

    The advection is in conservative form, :math:`(u^2/2)_x`, with a local Lax-Friedrichs flux on the interior facets
    and the interior trace on the exterior ones, which leaves the boundary transmissive. The viscous term is SIPG with
    natural boundaries. The initial value is :math:`\cos(2 \pi x)`.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    order : int, optional
        Degree of the elements.
    nu : float, optional
        Viscosity.
    sigma : float, optional
        Penalty constant, used only when `penalty` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`, see the module docstring.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, by default Newton with the tolerances of the FEniCS version.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(
        self, n=128, order=4, nu=0.02, sigma=10.0, penalty=None, mesh=None, solver_parameters=None, LHS_cache_size=12
    ):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.UnitIntervalMesh(n) if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'DG', order, variant='equispaced')
        alpha = sigma * max(order, 1) ** 2 if penalty is None else penalty
        n_facet = fd.FacetNormal(mesh)

        def mass(u, v):
            return u * v * fd.dx

        def rhs(w, v):
            flux = fd.as_vector([w**2 / 2])
            left, right = abs(w('+')), abs(w('-'))
            speed = fd.conditional(fd.gt(left, right), left, right)
            flux_hat = fd.avg(flux) + 0.5 * speed * fd.jump(w, n_facet)
            return (
                fd.dot(flux, fd.grad(v)) * fd.dx
                - fd.dot(flux_hat, fd.jump(v, n_facet)) * fd.dS
                - fd.dot(flux, n_facet) * v * fd.ds
                - sipg(w, v, nu, alpha)
            )

        x = fd.SpatialCoordinate(mesh)[0]
        if solver_parameters is None:
            solver_parameters = {'snes_atol': 1e-12, 'snes_rtol': 1e-11}
        super().__init__(
            V,
            mass,
            rhs,
            u_exact=fd.cos(2 * fd.pi * x),
            solver_parameters=solver_parameters,
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'nu', 'sigma', 'penalty', localVars=locals(), readOnly=True)

    def u_exact(self, t):
        """
        The initial value. There is no exact solution for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial value
        """
        assert t == 0, 'There is no exact solution of this Burgers problem, only the initial value at t = 0'
        return super().u_exact(t)


class GrayScott1DDGFiredrake(FormFiredrake):
    r"""
    Gray-Scott in 1D, DG in space. The port of
    :class:`~pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_FEniCS.fenics_grayscott_dg_mass`.

    Both diffusion terms are SIPG with natural boundaries, matching the zero-flux boundaries of
    :class:`~pySDC.implementations.problem_classes.GrayScott_1D_Firedrake.GrayScott1DFiredrake`; the reaction terms
    are the same.

    Parameters
    ----------
    n : int, optional
        Number of cells of :math:`[0, 100]`, ignored if `mesh` is given.
    order : int, optional
        Degree of the elements.
    Du, Dv, A, B : float, optional
        Diffusion, feed and decay rates, see ``GrayScott1DFiredrake``.
    sigma : float, optional
        Penalty constant, used only when `penalty` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`, see the module docstring.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 100]`.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, see ``FormFiredrake``.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(
        self,
        n=256,
        order=4,
        Du=1.0,
        Dv=0.01,
        A=0.09,
        B=0.086,
        sigma=10.0,
        penalty=None,
        mesh=None,
        solver_parameters=None,
        LHS_cache_size=12,
    ):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.IntervalMesh(n, 100) if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'DG', order, variant='equispaced')
        alpha = sigma * max(order, 1) ** 2 if penalty is None else penalty

        def mass(w, q):
            return fd.inner(w, q) * fd.dx

        def rhs(w, q):
            u, v = fd.split(w)
            q_u, q_v = fd.split(q)
            return (
                (-u * v**2 + A * (1 - u)) * q_u * fd.dx
                - sipg(u, q_u, Du, alpha)
                + (u * v**2 - B * v) * q_v * fd.dx
                - sipg(v, q_v, Dv, alpha)
            )

        x = fd.SpatialCoordinate(mesh)[0]
        bump = fd.sin(fd.pi * x / 100) ** 100
        super().__init__(
            V * V,
            mass,
            rhs,
            u_exact=fd.as_vector([1 - 0.5 * bump, 0.25 * bump]),
            solver_parameters=solver_parameters,
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister(
            'n', 'order', 'Du', 'Dv', 'A', 'B', 'sigma', 'penalty', localVars=locals(), readOnly=True
        )

    def u_exact(self, t):
        """
        The initial value. There is no exact solution of Gray-Scott for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial value
        """
        assert t == 0, 'There is no exact solution of Gray-Scott, only the initial value at t = 0'
        return super().u_exact(t)
