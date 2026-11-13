from pySDC.core.errors import DataError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.ufl_forms import implicit_system, total_rhs
from pySDC.implementations.datatype_classes.firedrake_mesh import (
    firedrake_mesh,
    firedrake_dual_mesh,
    IMEX_firedrake_dual_mesh,
)
import firedrake as fd
from firedrake.assemble import get_assembler


class FormFiredrake(Problem):
    r"""
    Generic problem given by UFL weak forms and solved with Firedrake, with the right-hand side treated implicitly.

    The semi-discrete system is

    .. math::
        M u' = R(u),

    where :math:`M` is assembled from the mass form :math:`m(u, v)` and :math:`R(u)` from the right-hand side form
    :math:`r(u, v)`, plus the explicit part :math:`r_E(u, v)` if one is given (this class treats it implicitly too,
    see :class:`FormFiredrakeIMEX` for splitting). Time enters only through the Firedrake ``Constant`` `t`, which is
    set to the requested time before every evaluation and solve, so forms, boundary values and the exact solution can
    all depend on it.

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

    Parameters
    ----------
    V : firedrake.FunctionSpace
        Function space of the solution
    mass : callable
        Mass form :math:`m(u, v)`
    rhs : callable
        Right-hand side form :math:`r(u, v)`, may be nonlinear in :math:`u`
    rhs_exp : callable, optional
        Explicit part :math:`r_E(u, v)` of the right-hand side
    bcs : tuple of firedrake.DirichletBC, optional
        Boundary conditions imposed in the solves. Their values may depend on `t`. Pass a tuple, not a list: pySDC reads
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
    """

    dtype_u = firedrake_mesh
    dtype_f = firedrake_dual_mesh
    fix_bc_for_residual = True

    def __init__(
        self,
        V,
        mass,
        rhs,
        rhs_exp=None,
        bcs=None,
        t=None,
        u_exact=None,
        solver_parameters=None,
        LHS_cache_size=12,
        stop_at_divergence=True,
    ):
        """Initialization, see the class docstring for the arguments."""
        self.V = V
        self.mass = mass
        self.rhs = rhs
        self.rhs_exp = rhs_exp
        self.bcs = () if bcs is None else bcs
        self.t = fd.Constant(0.0) if t is None else t
        self.u_exact_expr = u_exact
        self.solver_parameters = solver_parameters
        self.stop_at_divergence = stop_at_divergence

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
        return [total_rhs(self.rhs, self.rhs_exp)(self._u, self._v)]

    @property
    def implicit_rhs(self):
        """The part of the right-hand side the solves treat implicitly, as a callable (u, v) -> form."""
        return total_rhs(self.rhs, self.rhs_exp)

    def _assemble(self, assembler, u, out):
        """
        Assemble a form at `u`, without boundary conditions.

        Args:
            assembler: Firedrake assembler of a form in terms of ``self._u``
            u (dtype_u): Where to evaluate
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
            u (dtype_u): Solution at which to evaluate
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

        if factor not in self.solvers:
            if len(self.solvers) >= self.LHS_cache_size:
                self.solvers.pop(next(iter(self.solvers)))

            F, J = implicit_system(
                self.mass, self.implicit_rhs, self._x, self._v, fd.TrialFunction(self.V), fd.Constant(factor)
            )
            problem = fd.NonlinearVariationalProblem(F - self._b, self._x, bcs=self.bcs, J=J)
            self.solvers[factor] = fd.NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters)
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
        """Initialization, see :class:`FormFiredrake`. The explicit part `rhs_exp` is required here."""
        super().__init__(V, mass, rhs, rhs_exp, **kwargs)

    def make_rhs_forms(self):
        """
        Forms assembled by ``eval_f``, all evaluated at the buffer ``self._u``.

        Returns:
            list of ufl.Form: Implicit and explicit part
        """
        return [self.rhs(self._u, self._v), self.rhs_exp(self._u, self._v)]

    @property
    def implicit_rhs(self):
        """The part of the right-hand side the solves treat implicitly, as a callable (u, v) -> form."""
        return self.rhs

    def eval_f(self, u, t):
        """
        Evaluate the implicit and explicit right-hand sides at time `t`, without inverting the mass matrix.

        Args:
            u (dtype_u): Solution at which to evaluate
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
