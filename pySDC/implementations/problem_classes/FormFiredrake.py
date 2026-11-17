import math

from pySDC.core.errors import DataError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.ufl_forms import implicit_system
from pySDC.implementations.datatype_classes.firedrake_mesh import (
    firedrake_mesh,
    firedrake_dual_mesh,
    IMEX_firedrake_dual_mesh,
)
import firedrake as fd
from firedrake.assemble import get_assembler


def part_of(f, space):
    """
    The view of `f` on `space`: its own space, or a sub-space of it at any depth, such as ``W.sub(0)`` of a mixed space
    or ``V.sub(1)``, a component of a vector space. Writing to the view writes to `f`.

    Args:
        f (firedrake.Function): Function on the whole space
        space (firedrake.FunctionSpace): The space of `f` or one of its sub-spaces

    Returns:
        firedrake.Function: `f` on `space`
    """
    if space == f.function_space():
        return f
    return part_of(f, space.parent).sub(space.index if space.index is not None else space.component)


class FormFiredrake(Problem):
    r"""
    Generic problem given by UFL weak forms and solved with Firedrake, with the right-hand side treated implicitly.

    The semi-discrete system is

    .. math::
        M u' = R(u),

    where :math:`M` is assembled from the mass form :math:`m(u, v)` and :math:`R(u)` from the right-hand side form
    :math:`r(u, v)`, all of it treated implicitly (see :class:`FormFiredrakeIMEX` for an explicit part, and
    :func:`~pySDC.helpers.ufl_forms.total_rhs` for joining the parts of a split problem). Time enters only through
    the Firedrake ``Constant`` `t`, which is set to the requested time before every evaluation and solve, so forms,
    boundary values and the exact solution can all depend on it.

    This class never inverts :math:`M`: ``eval_f`` returns the load vector :math:`R(u)`, so it must be used with the
    mass-matrix sweepers :class:`~pySDC.implementations.sweeper_classes.generic_implicit_mass.generic_implicit_mass`
    and :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`.

    **Dual vectors.** :math:`R(u)`, :math:`M u`, the right-hand sides of the solves and the sweeper's residual are load
    vectors, which live in the dual space. They are stored as ``Cofunction`` objects (``dtype_f`` is
    :class:`firedrake_dual_mesh`), so adding one to a solution by mistake raises, and their ``abs`` is the norm dual to
    the :math:`L^2` norm, which the residual reports.

    **Dirichlet boundary conditions.** Nothing is assembled with ``bcs``, because Firedrake zeros boundary rows *and*
    columns then, while the collocation problem needs the full :math:`M u_0` and :math:`R(u)` on the rows of the
    interior nodes. The boundary values are imposed only by the solves, where Firedrake replaces the boundary rows by
    :math:`u = g(t)`. The boundary rows of the sweeper's residual therefore carry no equation; ``fix_residual`` zeros
    them (the sweepers call it because ``fix_bc_for_residual`` is set).

    **Several levels.** MLSDC and PFASST work with
    :class:`~pySDC.implementations.transfer_classes.BaseTransfer_mass.base_transfer_mass` and
    :class:`~pySDC.implementations.transfer_classes.TransferFiredrakeMesh.MeshToMeshFiredrakeHierarchy` on the
    meshes of a Firedrake ``MeshHierarchy``. Give `V`, `bcs` and `u_exact` per level, as lists in the problem
    parameters, since each lives on its level's mesh. On coarse levels, the solves impose the boundary values of the
    restricted fine solution instead of the data (``use_restricted_boundary_values``), so the coarse correction is
    zero on the boundary.

    Parameters
    ----------
    V : firedrake.FunctionSpace
        Function space of the solution
    mass : callable
        Mass form :math:`m(u, v)`
    rhs : callable
        Right-hand side form :math:`r(u, v)`, may be nonlinear in :math:`u`
    bcs : tuple of firedrake.DirichletBC, optional
        Boundary conditions imposed in the solves. Their values may depend on `t`, and they may act on a sub-space,
        such as one field of a mixed space or one component of a vector space. Pass a tuple, not a list: pySDC reads
        lists in the problem parameters as one entry per level.
    t : firedrake.Constant, optional
        The time the forms depend on. A new one is made if none is given.
    u_exact : UFL expression, optional
        Exact solution in terms of `t` and a spatial coordinate. Needed for ``u_exact``.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves. Defaults to Firedrake's (Newton with a direct linear solver).
    LHS_cache_size : int, optional
        Number of solvers kept, one per factor
    stop_at_divergence : bool, optional
        Raise if a solve does not converge, instead of logging it
    nullspace : firedrake.VectorSpaceBasis or firedrake.MixedVectorSpaceBasis, optional
        Null space of the implicit systems, e.g. the constants of a pressure or stream function with no Dirichlet
        condition. Passed on to the solves; a direct solver then needs null pivot detection, such as MUMPS with
        ``'mat_mumps_icntl_24': 1``.
    """

    dtype_u = firedrake_mesh
    dtype_f = firedrake_dual_mesh
    fix_bc_for_residual = True

    def __init__(
        self,
        V,
        mass,
        rhs,
        bcs=None,
        t=None,
        u_exact=None,
        solver_parameters=None,
        LHS_cache_size=12,
        stop_at_divergence=True,
        nullspace=None,
    ):
        """Initialization, see the class docstring for the arguments."""
        self.V = V
        self.mass = mass
        self.rhs = rhs
        self.bcs = () if bcs is None else bcs
        # the solves take their boundary values from this buffer: the data, or restricted values on coarse levels. A
        # condition on a sub-space reads the matching part of it.
        self._g = fd.Function(V)
        self._solve_bcs = tuple(bc.reconstruct(g=part_of(self._g, bc.function_space())) for bc in self.bcs)
        self._restricted = {}
        self.t = fd.Constant(0.0) if t is None else t
        self.u_exact_expr = u_exact
        self.solver_parameters = solver_parameters
        self.stop_at_divergence = stop_at_divergence
        self.nullspace = nullspace

        super().__init__(V)
        self._makeAttributeAndRegister('LHS_cache_size', localVars=locals(), readOnly=True)

        # buffers the cached forms and solvers are built on
        self._u = fd.Function(V)
        self._x = fd.Function(V)
        self._b = fd.Cofunction(V.dual())
        self._v = fd.TestFunction(V)

        # assemblers are cached: fd.assemble would redo the symbolic preprocessing on every call
        self._mass_assembler = get_assembler(mass(self._u, self._v))
        self._rhs_assemblers = [get_assembler(form) for form in self.make_rhs_forms()]
        self.solvers = {}

        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['ksp'] = WorkCounter()
        self.work_counters['solver_setup'] = WorkCounter()
        self.work_counters['solver'] = WorkCounter()

    def make_rhs_forms(self):
        """
        Forms assembled by ``eval_f``, all evaluated at the buffer ``self._u``.

        Returns:
            list of ufl.Form: The one form of the full right-hand side
        """
        return [self.rhs(self._u, self._v)]

    def _assemble(self, assembler, u, out):
        """
        Assemble a form at `u`, without boundary conditions.

        Args:
            assembler: Firedrake assembler of a form in terms of ``self._u``
            u (firedrake_mesh): Where to evaluate, a solution
            out (firedrake_dual_mesh): Where to put the result

        Returns:
            firedrake_dual_mesh: `out`
        """
        if u.is_dual:
            raise DataError('Forms are evaluated at a solution (a Function), not at a load vector (a Cofunction)')
        self._u.assign(u.functionspace)
        assembler.assemble(tensor=out.functionspace)
        return out

    def eval_f(self, u, t):
        """
        Evaluate the right-hand side :math:`R(u)` at time `t`, without inverting the mass matrix.

        Args:
            u (firedrake_mesh): Solution at which to evaluate
            t (float): Time at which to evaluate

        Returns:
            dtype_f: The load vector :math:`R(u)`
        """
        self.t.assign(t)
        self.work_counters['rhs']()
        return self._assemble(self._rhs_assemblers[0], u, self.dtype_f(self.init))

    def apply_mass_matrix(self, u):
        """
        Apply the mass matrix, without boundary conditions.

        Args:
            u (dtype_u): Function to apply it to

        Returns:
            firedrake_dual_mesh: :math:`M u`
        """
        return self._assemble(self._mass_assembler, u, firedrake_dual_mesh(self.init))

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`M u - \text{factor}\, R(u) = \text{rhs}` with the boundary conditions at time `t`.

        The Jacobian comes from ``ufl.derivative``. One Firedrake ``NonlinearVariationalSolver`` is set up per factor
        and cached; the oldest one is evicted once more than ``LHS_cache_size`` are around.

        Args:
            rhs (firedrake_dual_mesh): Right-hand side, a load vector
            factor (float): Prefactor of the implicit right-hand side, e.g. the node-to-node step size
            u0 (dtype_u): Initial guess
            t (float): Time of the solution

        Returns:
            dtype_u: The solution
        """
        self.t.assign(t)
        self._x.assign(u0.functionspace)
        # Cofunction.assign would silently take the entries of a Function
        if not rhs.is_dual:
            raise DataError('The right-hand side of the solve must be a load vector (a Cofunction), not a Function')
        self._b.assign(rhs.functionspace)

        # matched with a tolerance: the transfer and the sweeper compute the node times separately, and a miss would
        # silently impose the data instead, which stalls MLSDC
        restricted = next(
            (u for _t, u in self._restricted.items() if math.isclose(_t, t, rel_tol=1e-14, abs_tol=1e-14)), None
        )
        if restricted is None:
            for bc in self.bcs:
                bc.apply(self._g)
        else:
            self._g.assign(restricted.functionspace)

        if factor not in self.solvers:
            if len(self.solvers) >= self.LHS_cache_size:
                self.solvers.pop(next(iter(self.solvers)))

            F, J = implicit_system(self.mass, self.rhs, self._x, self._v, fd.TrialFunction(self.V), fd.Constant(factor))
            problem = fd.NonlinearVariationalProblem(F - self._b, self._x, bcs=self._solve_bcs, J=J)
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, nullspace=self.nullspace
            )
            self.work_counters['solver_setup']()

        try:
            self.solvers[factor].solve()
        except fd.exceptions.ConvergenceError as error:
            if self.stop_at_divergence:
                raise error
            self.logger.debug(error)

        self.work_counters['ksp'].niter += self.solvers[factor].snes.getLinearSolveIterations()
        self.work_counters['solver']()
        return self.dtype_u(self._x)

    def use_restricted_boundary_values(self, u, t):
        """
        Impose the boundary values of the restricted solution `u` in the solves at the times `t`, instead of the
        boundary data, see :meth:`pySDC.core.problem.Problem.use_restricted_boundary_values`. FAS calls this on
        coarse levels, so the coarse correction has homogeneous boundary values.

        Args:
            u (list of dtype_u): Restricted solution at the coarse nodes
            t (list of float): Times of the coarse nodes
        """
        self._restricted = {_t: self.dtype_u(_u) for _u, _t in zip(u, t, strict=True)}

    def fix_residual(self, res):
        """
        Zero the boundary rows of a residual. They hold no equation, since the solves replace them by the boundary
        values, see the class docstring.

        Args:
            res (dtype_u): Residual, changed in place
        """
        for bc in self.bcs:
            bc.zero(res.functionspace)

    def u_exact(self, t):
        """
        Interpolate the exact solution given at construction.

        Args:
            t (float): Time of the exact solution

        Returns:
            dtype_u: Exact solution
        """
        if self.u_exact_expr is None:
            raise NotImplementedError(f'{type(self).__name__} was set up without an exact solution')
        self.t.assign(t)
        me = self.u_init
        me.interpolate(self.u_exact_expr)
        return me


class FormFiredrakeIMEX(FormFiredrake):
    r"""
    Generic problem given by UFL weak forms and solved with Firedrake, IMEX: :math:`r` implicit, :math:`r_E` explicit.

    Everything else is as in :class:`FormFiredrake`; use it with
    :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`.
    """

    dtype_f = IMEX_firedrake_dual_mesh

    def __init__(self, V, mass, rhs, rhs_exp, **kwargs):
        """
        Initialization, see :class:`FormFiredrake` for the other arguments.

        Args:
            rhs (callable): Implicit part :math:`r(u, v)` of the right-hand side
            rhs_exp (callable): Explicit part :math:`r_E(u, v)` of the right-hand side
        """
        self.rhs_exp = rhs_exp
        super().__init__(V, mass, rhs, **kwargs)

    def make_rhs_forms(self):
        """
        Forms assembled by ``eval_f``, all evaluated at the buffer ``self._u``.

        Returns:
            list of ufl.Form: Implicit and explicit part
        """
        return [self.rhs(self._u, self._v), self.rhs_exp(self._u, self._v)]

    def eval_f(self, u, t):
        """
        Evaluate the implicit and explicit right-hand sides at time `t`, without inverting the mass matrix.

        Args:
            u (firedrake_mesh): Solution at which to evaluate
            t (float): Time at which to evaluate

        Returns:
            dtype_f: The load vectors, with `impl` and `expl` parts
        """
        self.t.assign(t)
        me = self.dtype_f(self.init)
        self._assemble(self._rhs_assemblers[0], u, me.impl)
        self._assemble(self._rhs_assemblers[1], u, me.expl)
        self.work_counters['rhs']()
        return me
