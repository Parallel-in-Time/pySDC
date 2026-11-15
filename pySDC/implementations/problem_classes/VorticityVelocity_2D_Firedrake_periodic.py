r"""
The 2D periodic vorticity problem with Firedrake, in the mass-matrix formulation, in two variants.

On the unit square with periodic boundaries, the vorticity :math:`\omega` is advected by the velocity it induces and
diffuses,

.. math::
    \omega_t = \nu \Delta \omega - u \cdot \nabla \omega, \quad u = (\psi_y, -\psi_x), \quad -\Delta \psi = \omega,

starting from a double shear layer. The stream function :math:`\psi` is a derived quantity, which the two variants
treat differently:

- :class:`Vortex2DFiredrake` solves the vorticity alone and computes :math:`\psi` by a Poisson solve in every
  right-hand side evaluation. This is the port of
  :class:`~pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic.fenics_vortex_2d_mass`.
- :class:`Vortex2DMixedFiredrake` makes :math:`\psi` an unknown next to :math:`\omega`, and the Poisson equation an
  algebraic equation of the system, given by forms like everything else.

They converge to the same collocation solution in the same SDC iterations; the mixed one costs more per iteration on
small meshes, since every vector operation handles both fields. On the periodic domain :math:`\psi` is only defined
up to a constant, which drops out of the velocity; the Poisson solves remove it with a null space.
"""

import firedrake as fd

from pySDC.implementations.datatype_classes.firedrake_mesh import riesz_map
from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX

#: a direct solver that copes with the constant null space of the periodic Poisson problem
MUMPS_NULL_PIVOTS = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_mumps_icntl_24': 1,
}


def double_shear_layer(mesh, rho, delta):
    """
    The initial vorticity of the FEniCS version, two shear layers of steepness `rho` with a perturbation of amplitude
    `delta`, summed over periodic images in y.

    Args:
        mesh (firedrake.MeshGeometry): The mesh
        rho (float): Steepness of the shear layers
        delta (float): Amplitude of the perturbation

    Returns:
        UFL expression: The initial vorticity
    """
    x, y = fd.SpatialCoordinate(mesh)

    def upper(k):
        return rho * (1 - fd.tanh(rho * ((0.75 + k) - y)) ** 2)

    def lower(k):
        return rho * (1 - fd.tanh(rho * (y - (0.25 + k))) ** 2)

    # as transcribed from the FEniCS version, whose first image enters with +; it lies at y = -3.25 and contributes
    # exp(-375) on the unit square
    w = upper(-4) + lower(-4)
    for k in range(-3, 5):
        w = w - upper(k) + lower(k)
    return w - delta * 2 * fd.pi * fd.cos(2 * fd.pi * (x + 0.25))


def advection(psi, w, v):
    r"""The advection form :math:`(\psi_y \omega_x - \psi_x \omega_y, v)`, i.e. :math:`-(u \cdot \nabla \omega, v)`."""
    return (psi.dx(1) * w.dx(0) - psi.dx(0) * w.dx(1)) * v * fd.dx


def poisson_solver(V, w, psi):
    r"""
    A cached solver for :math:`(\nabla \psi, \nabla q) = (\omega, q)` on the periodic space `V`, with the constants
    removed.

    Args:
        V (firedrake.FunctionSpace): The space
        w (firedrake.Function): The vorticity, read at every solve
        psi (firedrake.Function): Where the stream function goes

    Returns:
        firedrake.LinearVariationalSolver: The solver
    """
    p, q = fd.TrialFunction(V), fd.TestFunction(V)
    problem = fd.LinearVariationalProblem(fd.inner(fd.grad(p), fd.grad(q)) * fd.dx, w * q * fd.dx, psi)
    nullspace = fd.VectorSpaceBasis(constant=True, comm=V.mesh().comm)
    return fd.LinearVariationalSolver(problem, nullspace=nullspace, solver_parameters=MUMPS_NULL_PIVOTS)


class Vortex2DFiredrake(FormFiredrakeIMEX):
    r"""
    The periodic vorticity problem, with the stream function computed by a Poisson solve in every right-hand side
    evaluation; see the module docstring. Diffusion is implicit, advection explicit. This is the port of
    :class:`~pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic.fenics_vortex_2d_mass`.

    Parameters
    ----------
    n : int, optional
        Number of squares per direction, each split into two triangles, ignored if `mesh` is given.
    order : int, optional
        Degree of the Lagrange elements.
    nu : float, optional
        Viscosity :math:`\nu`.
    rho : float, optional
        Steepness of the shear layers in the initial value.
    delta : float, optional
        Amplitude of the perturbation in the initial value.
    mesh : firedrake.MeshGeometry, optional
        A periodic mesh of the unit square, e.g. from a ``MeshHierarchy``.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, n=32, order=2, nu=0.01, rho=50, delta=0.05, mesh=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        # `diagonal='right'` splits the squares like DOLFIN's UnitSquareMesh
        mesh = fd.PeriodicUnitSquareMesh(n, n, diagonal='right') if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'CG', order)
        self.psi = fd.Function(V)
        self._w = fd.Function(V)
        self._poisson = poisson_solver(V, self._w, self.psi)

        super().__init__(
            V,
            lambda w, v: w * v * fd.dx,
            lambda w, v: -nu * fd.inner(fd.grad(w), fd.grad(v)) * fd.dx,
            lambda w, v: advection(self.psi, w, v),
            u_exact=double_shear_layer(mesh, rho, delta),
            solver_parameters={'snes_type': 'ksponly'},
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'nu', 'rho', 'delta', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Solve for the stream function of `u`, then evaluate the right-hand side, see ``FormFiredrakeIMEX.eval_f``.

        Args:
            u (firedrake_mesh): Vorticity, or a load vector, which is evaluated at the function it represents
            t (float): Time

        Returns:
            dtype_f: The load vectors, with `impl` and `expl` parts
        """
        self._w.assign(riesz_map(self.V)(u.functionspace) if u.is_dual else u.functionspace)
        self._poisson.solve()
        return super().eval_f(u, t)

    def u_exact(self, t):
        """
        The initial value. There is no exact solution for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial vorticity
        """
        assert t == 0, 'There is no exact solution of the vortex problem, only the initial value at t = 0'
        return super().u_exact(t)


class Vortex2DMixedFiredrake(FormFiredrakeIMEX):
    r"""
    The periodic vorticity problem with the stream function as an unknown; see the module docstring. The unknowns are
    :math:`(\omega, \psi)` on a mixed space, the mass form acts on :math:`\omega` only, and the Poisson equation
    :math:`0 = (\omega, q) - (\nabla \psi, \nabla q)` is part of the implicit right-hand side, so it is solved with the
    diffusion. Advection is explicit. The constant mode of :math:`\psi` is removed by a null space on the solves.

    Parameters
    ----------
    n, order, nu, rho, delta, mesh, LHS_cache_size :
        As for :class:`Vortex2DFiredrake`.
    """

    def __init__(self, n=32, order=2, nu=0.01, rho=50, delta=0.05, mesh=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.PeriodicUnitSquareMesh(n, n, diagonal='right') if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'CG', order)
        W = V * V

        def mass(u, z):
            w, _ = fd.split(u)
            v, _ = fd.split(z)
            return w * v * fd.dx

        def implicit(u, z):
            w, psi = fd.split(u)
            v, q = fd.split(z)
            return (-nu * fd.inner(fd.grad(w), fd.grad(v)) + w * q - fd.inner(fd.grad(psi), fd.grad(q))) * fd.dx

        def explicit(u, z):
            w, psi = fd.split(u)
            v, _ = fd.split(z)
            return advection(psi, w, v)

        super().__init__(
            W,
            mass,
            implicit,
            explicit,
            solver_parameters={'snes_type': 'ksponly', **MUMPS_NULL_PIVOTS},
            nullspace=fd.MixedVectorSpaceBasis(W, [W.sub(0), fd.VectorSpaceBasis(constant=True, comm=mesh.comm)]),
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'nu', 'rho', 'delta', localVars=locals(), readOnly=True)
        self._initial_vorticity = double_shear_layer(mesh, rho, delta)
        self._V = V

    def u_exact(self, t):
        """
        The initial value: the vorticity and its stream function. There is no exact solution for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial vorticity and stream function
        """
        assert t == 0, 'There is no exact solution of the vortex problem, only the initial value at t = 0'
        me = self.u_init
        w, psi = me.functionspace.subfunctions
        w.interpolate(self._initial_vorticity)
        poisson_solver(self._V, w, psi).solve()
        return me
