from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh
from gusto.core.labels import (
    time_derivative,
    implicit,
    explicit,
)
from firedrake.fml import replace_subject, all_terms, drop
import firedrake as fd
import numpy as np


class GenericGusto(Problem):
    """
    Set up solvers based on the equation. Keep in mind that you probably want to use the pySDC-Gusto coupling via
    the `pySDC_integrator` class in the helpers in order to get spatial methods rather than interfacing with this
    class directly.

    Gusto equations work by a residual, which is minimized in nonlinear solvers to obtain the right hand side
    evaluation or the solution to (IMEX) Euler steps. You control what you solve for by manipulating labeled parts
    of the residual.
    """

    dtype_u = firedrake_mesh
    dtype_f = firedrake_mesh
    rhs_n_labels = 1

    def __init__(
        self,
        equation,
        apply_bcs=True,
        solver_parameters=None,
        stop_at_divergence=False,
        LHS_cache_size=12,
        residual=None,
        *active_labels,
    ):
        """
        Initialisation

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            solver_params (dict, optional): Solver parameters for the nonlinear variational problems
            stop_at_divergence (bool, optional): Whether to raise an error when the variational problems do not converge. Defaults to False
            LHS_cache_size (int, optional): Size of the cache for solvers. Defaults to 12.
            residual (Firedrake.form, optional): Overwrite the residual of the equation, e.g. after adding spatial methods. Defaults to None.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """

        self.equation = equation
        self.residual = equation.residual if residual is None else residual
        self.field_name = equation.field_name
        self.fs = equation.function_space
        self.idx = None
        if solver_parameters is None:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        self.solver_parameters = solver_parameters
        self.stop_at_divergence = stop_at_divergence

        # -------------------------------------------------------------------- #
        # Setup caches
        # -------------------------------------------------------------------- #

        self.x_out = fd.Function(self.fs)
        self.solvers = {}
        self._u = fd.Function(self.fs)

        super().__init__(self.fs)
        self._makeAttributeAndRegister('LHS_cache_size', 'apply_bcs', localVars=locals(), readOnly=True)
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['ksp'] = WorkCounter()
        self.work_counters['solver_setup'] = WorkCounter()
        self.work_counters['solver'] = WorkCounter()

    @property
    def bcs(self):
        if not self.apply_bcs:
            return None
        else:
            return self.equation.bcs[self.equation.field_name]

    def invert_mass_matrix(self, rhs):
        self._u.assign(rhs.functionspace)

        if 'mass_matrix' not in self.solvers.keys():
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )
            rhs_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self._u, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form - rhs_form).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers['mass_matrix'] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        self.solvers['mass_matrix'].solve()

        return self.dtype_u(self.x_out)

    def eval_f(self, u, *args):
        self._u.assign(u.functionspace)

        if 'eval_rhs' not in self.solvers.keys():
            residual = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=replace_subject(self._u, old_idx=self.idx),
                map_if_true=drop,
            )
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form + residual).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers['eval_rhs'] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        self.solvers['eval_rhs'].solve()
        self.work_counters['rhs']()

        return self.dtype_f(self.x_out)

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size + self.rhs_n_labels:
                self.solvers.pop(
                    [me for me in self.solvers.keys() if type(me) in [float, int, np.float64, np.float32]][0]
                )

            # setup left hand side (M - factor*f)(u)
            # put in output variable
            residual = self.residual.label_map(all_terms, map_if_true=replace_subject(self.x_out, old_idx=self.idx))
            # multiply f by factor
            residual = residual.label_map(
                lambda t: t.has_label(time_derivative), map_if_false=lambda t: fd.Constant(factor) * t
            )

            # subtract right hand side
            mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_false=drop)
            residual -= mass_form.label_map(all_terms, map_if_true=replace_subject(self._u, old_idx=self.idx))

            # construct solver
            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup']()

        try:
            self.solvers[factor].solve()
        except fd.exceptions.ConvergenceError as error:
            if self.stop_at_divergence:
                raise error
            else:
                self.logger.debug(error)

        self.work_counters['ksp'].niter += self.solvers[factor].snes.getLinearSolveIterations()
        self.work_counters['solver']()
        return self.dtype_u(self.x_out)


class GenericGustoImex(GenericGusto):
    dtype_f = IMEX_firedrake_mesh
    rhs_n_labels = 2

    def evaluate_labeled_term(self, u, label):
        self._u.assign(u.functionspace)

        if label not in self.solvers.keys():
            residual = self.residual.label_map(
                lambda t: t.has_label(label) and not t.has_label(time_derivative),
                map_if_true=replace_subject(self._u, old_idx=self.idx),
                map_if_false=drop,
            )
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )

            problem = fd.NonlinearVariationalProblem((mass_form + residual).form, self.x_out, bcs=self.bcs)
            solver_name = self.field_name + self.__class__.__name__
            self.solvers[label] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup'] = WorkCounter()

        self.solvers[label].solve()
        return self.x_out

    def eval_f(self, u, *args):
        me = self.dtype_f(self.init)
        me.impl.assign(self.evaluate_labeled_term(u, implicit))
        me.expl.assign(self.evaluate_labeled_term(u, explicit))
        self.work_counters['rhs']()
        return me

    def solve_system(self, rhs, factor, u0, *args):
        self.x_out.assign(u0.functionspace)  # set initial guess
        self._u.assign(rhs.functionspace)

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size + self.rhs_n_labels:
                self.solvers.pop(
                    [me for me in self.solvers.keys() if type(me) in [float, int, np.float64, np.float32]][0]
                )

            # setup left hand side (M - factor*f_I)(u)
            # put in output variable
            residual = self.residual.label_map(
                lambda t: t.has_label(time_derivative) or t.has_label(implicit),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop,
            )
            # multiply f_I by factor
            residual = residual.label_map(
                lambda t: t.has_label(implicit) and not t.has_label(time_derivative),
                map_if_true=lambda t: fd.Constant(factor) * t,
            )

            # subtract right hand side
            mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_false=drop)
            residual -= mass_form.label_map(all_terms, map_if_true=replace_subject(self._u, old_idx=self.idx))

            # construct solver
            problem = fd.NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)
            solver_name = f'{self.field_name}-{self.__class__.__name__}-{factor}'
            self.solvers[factor] = fd.NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters, options_prefix=solver_name
            )
            self.work_counters['solver_setup'] = WorkCounter()

        self.solvers[factor].solve()
        try:
            self.solvers[factor].solve()
        except fd.exceptions.ConvergenceError as error:
            if self.stop_at_divergence:
                raise error
            else:
                self.logger.debug(error)

        self.work_counters['ksp'].niter += self.solvers[factor].snes.getLinearSolveIterations()
        self.work_counters['solver']()
        return self.dtype_u(self.x_out)
