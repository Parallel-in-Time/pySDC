import firedrake as fd

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake


class Burgers1DFiredrake(FormFiredrake):
    r"""
    Viscous Burgers equation in one dimension with Firedrake, given by its weak forms. This is the port of
    :class:`~pySDC.projects.FEM_with_FEniCS.problem_classes.Burgers_1D_FEniCS.fenics_burgers_mass`, on the same mesh.

    .. math::
        \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}

    on :math:`[0, 1]` with natural (homogeneous Neumann) boundaries, so the weak right-hand side is
    :math:`-(u u_x, v) - \nu (u_x, v_x)` and there are no boundary conditions to impose. Everything else comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    order : int, optional
        Degree of the Lagrange elements.
    nu : float, optional
        Viscosity.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 1]`, e.g. from a ``MeshHierarchy`` for several levels.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, by default Newton with the tolerances of the FEniCS version.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, n=128, order=4, nu=0.02, mesh=None, solver_parameters=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.UnitIntervalMesh(n) if mesh is None else mesh
        # equispaced nodes are where FEniCS puts them, so the interpolated initial value is the same
        V = fd.FunctionSpace(mesh, 'CG', order, variant='equispaced')

        def mass(u, v):
            return u * v * fd.dx

        def rhs(u, v):
            return (-u * u.dx(0) * v - nu * u.dx(0) * v.dx(0)) * fd.dx

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
        self._makeAttributeAndRegister('n', 'order', 'nu', localVars=locals(), readOnly=True)

    def u_exact(self, t):
        r"""
        The initial value :math:`u(x, 0) = \cos(2 \pi x)`. There is no exact solution for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial value
        """
        assert t == 0, 'There is no exact solution of this Burgers problem, only the initial value at t = 0'
        return super().u_exact(t)
