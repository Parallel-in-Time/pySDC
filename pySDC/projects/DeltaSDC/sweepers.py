r"""
Deferred-correction ("delta-form") SDC sweepers.

A standard SDC sweep

.. math::
    u^{k+1}_m = u_0 + \tau_m + \Delta t (Q f^k)_m
                + \Delta t \sum_j Q^\Delta_{mj}\,(f^{k+1}_j - f^k_j)

is algebraically identical to, with :math:`\delta_m = u^{k+1}_m - u^k_m` and the collocation
residual :math:`\varepsilon_m = u_0 + \tau_m + \Delta t (Q f^k)_m - u^k_m`,

.. math::
    \delta_m = \varepsilon_m + \Delta t \sum_j Q^\Delta_{mj}\,\Delta f_j,
    \qquad \Delta f_j = f(u^k_j + \delta_j) - f(u^k_j).

Written this way, every sweep is iterative refinement: a high-precision residual, a correction
solve, and a high-precision update ``u <- u + delta``. No Jacobian appears, so an IMEX splitting
survives unchanged.

The point of the reformulation is that the quantity handed to the node-local solver is a
*correction*. Its magnitude tends to zero as the sweeps converge, so a reduced-precision solve
introduces an error proportional to :math:`|\delta|` rather than to :math:`|u|` and therefore does
not cap the attainable accuracy.

Three node-local strategies are supported, selected automatically:

``solve_system_delta``
    Used when the problem provides it. Solves
    :math:`\delta - \alpha[f(w+\delta) - f(w)] = r` for the correction directly. This is the only
    option for a nonlinear implicit operator, and the only one that hands a reduced-precision
    solver a small unknown.

``linear_implicit=True``
    For a linear or affine implicit operator, :math:`f(w+\delta) - f(w) = A\delta`, so the stock
    ``solve_system`` already solves the correction equation once the affine part
    :math:`\alpha f(0, t)` is removed from the right-hand side. No problem class needs changing.

fallback
    Otherwise the substitution :math:`y = u^k_m + \delta_m` reduces the correction equation to the
    ordinary implicit solve. Always correct and identical to :class:`generic_implicit`, but the
    solver sees an :math:`\mathcal{O}(1)` unknown, so there is no precision benefit.

``correction_precision`` additionally stores the small quantities
(:math:`\varepsilon`, :math:`\delta`, :math:`\Delta f`) in a reduced-precision datatype built from
the problem's own ``init`` tuple.
"""

import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class DeltaFormMixin:
    """Shared machinery for the delta-form sweepers."""

    def _delta_setup(self):
        """Read the optional sweeper parameters controlling the delta form."""
        token = getattr(self.params, 'correction_precision', None)
        self._work_dtype = None if token is None else np.dtype(token)
        self._linear_implicit = bool(getattr(self.params, 'linear_implicit', False))

    def _work_init(self, prob):
        """Build the problem's ``init`` tuple with the correction dtype substituted."""
        return (prob.init[0], prob.init[1], self._work_dtype)

    def _to_work(self, prob, value):
        """Store a small correction quantity in a reduced-precision datatype."""
        if self._work_dtype is None:
            return value
        me = prob.dtype_u(self._work_init(prob))
        me[:] = value
        return me

    def _to_backend(self, prob, value):
        """Lift a possibly reduced-precision quantity back to backend precision."""
        me = prob.dtype_u(prob.init)
        me[:] = value
        return me

    def _coeff(self, value):
        """Cast a scalar coefficient so an accumulation stays at the correction precision."""
        if self._work_dtype is None:
            return value
        return self._work_dtype.type(value)

    def _residual_nodes(self):
        r"""
        Compute :math:`\varepsilon_m = u_0 + \tau_m + \Delta t (Q f^k)_m - u^k_m`.

        This is the high-precision residual of iterative refinement. It is a difference of
        :math:`\mathcal{O}(1)` quantities and is therefore always formed in backend precision.

        Returns
        -------
        list
            One residual per collocation node.
        """
        lvl = self.level
        eps = self.integrate()
        for m in range(self.coll.num_nodes):
            eps[m] += lvl.u[0]
            eps[m] -= lvl.u[m + 1]
            if lvl.tau[m] is not None:
                eps[m] += lvl.tau[m]
        return eps

    def _solve_correction(self, rhs_corr, alpha, u_old, f_old, t_node, implicit_part=None):
        r"""
        Solve the node-local correction equation.

        Parameters
        ----------
        rhs_corr : dtype_u
            Right-hand side :math:`r` of the correction equation.
        alpha : float
            Implicit prefactor :math:`\alpha = \Delta t Q^\Delta_{mm}`.
        u_old : dtype_u
            Current nodal value :math:`u^k_m`, the base state of the correction.
        f_old : dtype_f
            ``f`` evaluated at ``u_old``; already stored on the level, so it costs nothing.
        t_node : float
            Physical time of the collocation node.
        implicit_part : dtype_u, optional
            The implicit component of ``f_old`` for IMEX problems. Defaults to ``f_old``.

        Returns
        -------
        dtype_u
            The correction :math:`\delta_m`.
        """
        prob = self.level.prob
        f_impl_old = f_old if implicit_part is None else implicit_part

        if alpha == 0:
            return self._to_backend(prob, rhs_corr)

        if hasattr(prob, 'solve_system_delta'):
            return prob.solve_system_delta(self._to_backend(prob, rhs_corr), alpha, u_old, f_old, t_node)

        rhs_phys = self._to_backend(prob, rhs_corr)
        zero = prob.dtype_u(prob.init, val=0.0)

        if self._linear_implicit:
            # f(w+d) - f(w) = A d, so solve_system already solves the correction equation once the
            # affine part f(0, t) has been removed. f(0, t) vanishes for a homogeneous operator.
            affine = prob.eval_f(zero, t_node)
            rhs_phys -= alpha * (affine if implicit_part is None else affine.impl)
            return prob.solve_system(rhs_phys, alpha, zero, t_node)

        # Fallback: substitute y = u_old + delta. Always correct, but the solver sees an O(1)
        # unknown, so there is no precision benefit.
        rhs_phys += u_old
        rhs_phys -= alpha * f_impl_old
        solution = prob.solve_system(rhs_phys, alpha, u_old, t_node)
        delta = prob.dtype_u(solution)
        delta -= u_old
        return delta


class delta_implicit(DeltaFormMixin, generic_implicit):
    """
    Delta-form counterpart of :class:`generic_implicit`.

    Mathematically identical to the standard sweep; see the module docstring for the sweeper
    parameters ``correction_precision`` and ``linear_implicit``.
    """

    def update_nodes(self):
        """
        Perform one delta-form sweep over all collocation nodes.

        Returns
        -------
        None
        """
        lvl = self.level
        prob = lvl.prob
        assert lvl.status.unlocked
        num_nodes = self.coll.num_nodes
        self._delta_setup()

        eps = [self._to_work(prob, value) for value in self._residual_nodes()]
        df = [None] * (num_nodes + 1)

        for m in range(num_nodes):
            t_node = lvl.time + lvl.dt * self.coll.nodes[m]

            rhs_corr = type(eps[m])(eps[m])
            for j in range(1, m + 1):
                if self.QI[m + 1, j] != 0.0:
                    rhs_corr += self._coeff(lvl.dt * self.QI[m + 1, j]) * df[j]

            alpha = lvl.dt * self.QI[m + 1, m + 1]
            u_old = prob.dtype_u(lvl.u[m + 1])
            f_old = prob.dtype_f(lvl.f[m + 1])

            delta = self._solve_correction(rhs_corr, alpha, u_old, f_old, t_node)

            lvl.u[m + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
            lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], t_node)

            increment = prob.dtype_u(prob.init)
            increment[:] = lvl.f[m + 1]
            increment -= f_old
            df[m + 1] = self._to_work(prob, increment)

        lvl.status.updated = True
        return None


class delta_imex_1st_order(DeltaFormMixin, imex_1st_order):
    """
    Delta-form counterpart of :class:`imex_1st_order`.

    The correction equation contains only differences of ``f``, never a Jacobian, so the
    explicit/implicit splitting is untouched.
    """

    def update_nodes(self):
        """
        Perform one delta-form IMEX sweep over all collocation nodes.

        ``QE`` is strictly lower triangular, which :class:`imex_1st_order` already enforces, so the
        explicit part never contributes to the node-local solve.

        Returns
        -------
        None
        """
        lvl = self.level
        prob = lvl.prob
        assert lvl.status.unlocked
        num_nodes = self.coll.num_nodes
        self._delta_setup()

        eps = [self._to_work(prob, value) for value in self._residual_nodes()]
        df_impl = [None] * (num_nodes + 1)
        df_expl = [None] * (num_nodes + 1)

        for m in range(num_nodes):
            t_node = lvl.time + lvl.dt * self.coll.nodes[m]

            rhs_corr = type(eps[m])(eps[m])
            for j in range(1, m + 1):
                if self.QI[m + 1, j] != 0.0:
                    rhs_corr += self._coeff(lvl.dt * self.QI[m + 1, j]) * df_impl[j]
                if self.QE[m + 1, j] != 0.0:
                    rhs_corr += self._coeff(lvl.dt * self.QE[m + 1, j]) * df_expl[j]

            alpha = lvl.dt * self.QI[m + 1, m + 1]
            u_old = prob.dtype_u(lvl.u[m + 1])
            f_old = prob.dtype_f(lvl.f[m + 1])

            delta = self._solve_correction(rhs_corr, alpha, u_old, f_old, t_node, implicit_part=f_old.impl)

            lvl.u[m + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
            lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], t_node)

            impl = prob.dtype_u(prob.init)
            impl[:] = lvl.f[m + 1].impl
            impl -= f_old.impl
            df_impl[m + 1] = self._to_work(prob, impl)

            expl = prob.dtype_u(prob.init)
            expl[:] = lvl.f[m + 1].expl
            expl -= f_old.expl
            df_expl[m + 1] = self._to_work(prob, expl)

        lvl.status.updated = True
        return None
