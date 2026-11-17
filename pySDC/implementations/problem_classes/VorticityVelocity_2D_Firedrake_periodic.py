r"""
The 2D periodic vorticity problem with Firedrake, in the mass-matrix formulation.

On the unit square with periodic boundaries, the vorticity :math:`\omega` is advected by the velocity it induces and
diffuses,

.. math::
    \omega_t = \nu \Delta \omega - u \cdot \nabla \omega, \quad u = (\psi_y, -\psi_x), \quad -\Delta \psi = \omega,

starting from a double shear layer. :class:`Vortex2DFiredrake` solves for the vorticity and computes the stream
function :math:`\psi` by a Poisson solve in every right-hand side evaluation. On the periodic domain :math:`\psi` is only defined up to a
constant, which drops out of the velocity; the Poisson solves remove it with a null space.
"""

import firedrake as fd

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
    The initial vorticity, two shear layers of steepness `rho` with a perturbation of amplitude `delta`, summed over
    periodic images in y.

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

    w = sum(lower(k) - upper(k) for k in range(-4, 5))
    return w - delta * 2 * fd.pi * fd.cos(2 * fd.pi * (x + 0.25))


def advection(psi, w, v):
    r"""
    The advection form :math:`-(u \cdot \nabla \omega, v) = (\psi_x \omega_y - \psi_y \omega_x, v)` with
    :math:`u = (\psi_y, -\psi_x)`. (pySDC's earlier version of this problem had the opposite sign, which advects the
    vorticity with :math:`-u`, a velocity it does not induce.)
    """
    return (psi.dx(0) * w.dx(1) - psi.dx(1) * w.dx(0)) * v * fd.dx


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
    # the matrix never changes, so it is assembled and factorized once
    problem = fd.LinearVariationalProblem(
        fd.inner(fd.grad(p), fd.grad(q)) * fd.dx, w * q * fd.dx, psi, constant_jacobian=True
    )
    nullspace = fd.VectorSpaceBasis(constant=True, comm=V.mesh().comm)
    return fd.LinearVariationalSolver(problem, nullspace=nullspace, solver_parameters=MUMPS_NULL_PIVOTS)


class Vortex2DFiredrake(FormFiredrakeIMEX):
    r"""
    The periodic vorticity problem, with the stream function computed by a Poisson solve in every right-hand side
    evaluation; see the module docstring. Diffusion is implicit, advection explicit.

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
        mesh = fd.PeriodicUnitSquareMesh(n, n) if mesh is None else mesh
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
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'nu', 'rho', 'delta', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Solve for the stream function of `u`, then evaluate the right-hand side, see ``FormFiredrakeIMEX.eval_f``.

        Args:
            u (firedrake_mesh): Vorticity
            t (float): Time

        Returns:
            dtype_f: The load vectors, with `impl` and `expl` parts
        """
        self._w.assign(u.functionspace)
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
