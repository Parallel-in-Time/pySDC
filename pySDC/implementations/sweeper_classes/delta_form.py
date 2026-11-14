r"""
Deferred-correction ("delta-form") SDC sweepers.

A standard SDC sweep

.. math::
    u^{k+1}_m = u_0 + \tau_m + \Delta t (Q f^k)_m
                + \Delta t \sum_j Q^\Delta_{mj}\,(f^{k+1}_j - f^k_j)

is algebraically identical to, with :math:`\delta_m = u^{k+1}_m - u^k_m` and the collocation
residual :math:`r_m = u_0 + \tau_m + \Delta t (Q f^k)_m - u^k_m`,

.. math::
    \delta_m = r_m + \Delta t \sum_j Q^\Delta_{mj}\,\Delta f_j,
    \qquad \Delta f_j = f(u^k_j + \delta_j) - f(u^k_j).

Written this way, every sweep is iterative refinement: a high-precision residual, a correction
solve, and a high-precision update ``u <- u + delta``. No Jacobian appears, so an IMEX splitting
survives unchanged.

The point of the reformulation is that the quantity handed to the node-local solver is a
*correction*. Its magnitude tends to zero as the sweeps converge, so a reduced-precision solve
introduces an error proportional to :math:`|\delta|` rather than to :math:`|u|` and therefore does
not cap the attainable accuracy. What the sweeper and the solver guarantee each other for that to
hold is the correction-solve contract, documented on :class:`~pySDC.core.problem.Problem`.

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
    The right-hand side is scaled to unit size around the call, so the solver may work in any
    precision without the shrinking correction falling out of its format's range.

fallback
    Otherwise the substitution :math:`y = u^k_m + \delta_m` reduces the correction equation to the
    ordinary implicit solve. Always correct and identical to
    :class:`~pySDC.implementations.sweeper_classes.generic_implicit.generic_implicit`, but the solver sees an
    :math:`\mathcal{O}(1)` unknown, so there is no precision benefit.

``correction_precision`` additionally stores the small quantities
(:math:`r`, :math:`\delta`, :math:`\Delta f`) in a reduced-precision datatype built from
the problem's own ``init`` tuple, scaled by the residual's magnitude so the format's mantissa is
used however small the correction gets.

The increment :math:`\Delta f` is formed by subtracting two stored right-hand sides unless the
problem provides ``eval_f_increment(base, delta, t)``, which expands it analytically. That
subtraction is a cancellation carrying the operator norm, so it binds as soon as a level runs below
backend precision.

The same sweeper serves any number of levels, and which it is doing is decided by the transfer, not
by the choice of sweeper. On the finest level it computes its own residual, which is where the
accuracy of the whole iteration is set. Given a transfer that hands one down -- by setting
``eps_in`` on the level's sweeper -- it uses that instead and advances it in place, so nothing is
ever rebuilt out of :math:`\mathcal{O}(1)` coarse state. :class:`~pySDC.core.base_transfer.BaseTransfer` hands
nothing down, so on a stock hierarchy this sweeper reproduces MLSDC and PFASST exactly.
"""

import sys

import numpy as np

from pySDC.core.errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

_ROW_KERNELS = {}


def _row_kernel(num_nodes, split, residual, tau):
    r"""
    A CuPy kernel for one row of :math:`\Delta t\, Q F`, optionally completed to the residual
    :math:`u_0 + \tau_m + \Delta t\, (Q F)_m - u_m`.

    It performs the operations of the loops in :meth:`generic_implicit.integrate`,
    :meth:`imex_1st_order.integrate` and :meth:`DeltaFormMixin._residual_nodes` one by one and in the
    same order, with contraction into fused multiply-adds switched off, so its result is bitwise
    theirs. It only reads every operand once, where the loops allocate and pass over a temporary for
    every term; on a GPU, these passes over memory are the time.
    """
    key = (num_nodes, split, residual, tau)
    if key not in _ROW_KERNELS:
        import cupy as cp

        if split:  # imex_1st_order sums the two parts first
            arrays = [name for j in range(num_nodes) for name in (f'i{j}', f'e{j}')]
            terms = [f'(i{j} + e{j})' for j in range(num_nodes)]
        else:
            arrays = [f'f{j}' for j in range(num_nodes)]
            terms = list(arrays)  # a copy: u0, um and tau are operands, not terms
        arrays += ['u0', 'um'] * residual + ['tau'] * tau
        params = ', '.join([f'float64 {name}' for name in arrays] + [f'float64 c{j}' for j in range(num_nodes)])
        body = 'double t = 0.0;' + ''.join(f' t = t + c{j} * {term};' for j, term in enumerate(terms))
        body += ' t = t + u0; t = t - um;' * residual + ' t = t + tau;' * tau
        _ROW_KERNELS[key] = cp.ElementwiseKernel(
            params,
            'float64 out',
            body + ' out = t;',
            f'delta_form_row_{num_nodes}_{split:d}{residual:d}{tau:d}',
            options=('--fmad=false',),
        )
    return _ROW_KERNELS[key]


def _gpu_rows(sweeper, residual):
    r"""
    :math:`\Delta t\, Q F` at every node, or the residual, in one GPU kernel per node.

    Returns ``None`` unless every operand is a ``float64`` CuPy array, in which case the caller takes
    its loop. A split right-hand side (``impl`` and ``expl``) is summed as
    :meth:`imex_1st_order.integrate` sums it.
    """
    cp = sys.modules.get('cupy')
    if cp is None:
        return None
    lvl = sweeper.level
    num_nodes = sweeper.coll.num_nodes
    f = lvl.f[1 : num_nodes + 1]
    split = hasattr(f[0], 'impl')
    terms = [part for value in f for part in ((value.impl, value.expl) if split else (value,))]
    operands = []
    for m in range(1, num_nodes + 1):
        extra = [lvl.u[0], lvl.u[m]] + [lvl.tau[m - 1]] * (lvl.tau[m - 1] is not None) if residual else []
        operands.append(terms + extra)
    if not all(isinstance(a, cp.ndarray) and a.dtype == cp.float64 for row in operands for a in row):
        return None
    rows = []
    for m, row in enumerate(operands, start=1):
        coefficients = [lvl.dt * sweeper.coll.Qmat[m, j] for j in range(1, num_nodes + 1)]
        tau = residual and lvl.tau[m - 1] is not None
        out = lvl.prob.dtype_u(lvl.prob.init)
        _row_kernel(num_nodes, split, residual, tau)(*row, *coefficients, out)
        rows.append(out)
    return rows


class DeltaFormMixin:
    """Shared machinery for the delta-form sweepers."""

    def _delta_setup(self):
        """Read the optional sweeper parameters, and arm the per-sweep recorders."""
        token = getattr(self.params, 'correction_precision', None)
        self._work_dtype = None if token is None else np.dtype(token)
        init = self.level.prob.init
        if (
            self._work_dtype is not None
            and self._work_dtype.kind != 'c'
            and isinstance(init, tuple)
            and np.dtype(init[-1]).kind == 'c'
        ):
            # storing a complex correction in a real array keeps only its real part
            raise ParameterError(
                f'correction_precision={self._work_dtype} is real, but the problem is complex ({np.dtype(init[-1])}). '
                'Use a complex correction_precision such as complex64.'
            )
        self._linear_implicit = bool(getattr(self.params, 'linear_implicit', False))
        self._deltas, self._dfs = [], []

    eps_in = None
    """Residual handed down by the transfer, or ``None`` on the finest level."""

    delta_acc = None
    """Corrections this level has accumulated since the last restriction."""

    def sync_initial_value(self):
        r"""
        Follow a change of :math:`u_0` made after the residual was handed down.

        PFASST receives the initial value from the predecessor *after* the restriction, directly
        into ``u[0]``. The residual depends on it additively, so following it is one addition:
        :math:`r_m \leftarrow r_m + (u_0 - u_0^{\mathrm{ref}})`. Exactly zero
        when nothing arrived, which is every serial run.

        This is the one place the hierarchy still differences two :math:`\mathcal{O}(1)` values, and
        it is the reason a reduced-precision coarse level buys less under PFASST than under MLSDC:
        the difference is small -- it is the coarse-versus-fine discrepancy at the step interface,
        and it converges to zero -- but it is *formed* by cancelling two values of size
        :math:`|u|`. Putting the step-to-step exchange itself in delta form is what would remove it.

        Returns
        -------
        None
        """
        lvl = self.level
        if self.eps_in is None or lvl.u0_reference is None:
            return
        shift = lvl.u[0] - lvl.u0_reference
        self.eps_in = [eps + shift for eps in self.eps_in]
        lvl.u0_reference = lvl.prob.dtype_u(lvl.u[0])
        return None

    def compute_residual(self, stage=''):
        r"""
        Report the residual this level is already tracking, instead of rebuilding one.

        A level with an inherited residual carries it forward through every sweep and prolongation,
        so recomputing it from :math:`\mathcal{O}(1)` state would be both redundant and less
        accurate. It is also what makes the :math:`\tau` term unnecessary: the only other reader is
        :meth:`compute_end_point`, and only when the end point comes from the quadrature update.

        Parameters
        ----------
        stage : str
            The stage of the step this level belongs to.

        Returns
        -------
        None
        """
        if self.eps_in is None:
            return super().compute_residual(stage=stage)

        lvl = self.level
        if stage in self.params.skip_residual_computation:
            lvl.status.residual = 0.0 if lvl.status.residual is None else lvl.status.residual
            return None

        lvl.residual = [lvl.prob.dtype_u(eps) for eps in self.eps_in]
        norms = [abs(eps) for eps in self.eps_in]
        kind = lvl.params.residual_type
        if kind not in ('full_abs', 'last_abs', 'full_rel', 'last_rel'):
            raise NotImplementedError(f'residual type "{kind}" not implemented!')
        value = norms[-1] if kind.startswith('last') else max(norms)
        lvl.status.residual = value / abs(lvl.u[0]) if kind.endswith('rel') else value
        lvl.status.updated = False
        return None

    def advance_residual(self, eps, deltas, dfs):
        r"""
        Advance a residual by an update: :math:`r \leftarrow r - \delta
        + \Delta t (Q \Delta f)`.

        Every term is small, so this never cancels. It is exact for any update to the nodal values,
        which is why it serves both a sweep and a prolongation.

        Parameters
        ----------
        eps : list
            The residual, one entry per node.
        deltas : list
            The update applied to the nodal values, one entry per node.
        dfs : list
            The resulting right-hand side increments, one entry per node.

        Returns
        -------
        list
            The advanced residual.
        """
        dt, Q = self.level.dt, self.coll.Qmat
        out = []
        for m in range(len(eps)):
            acc = eps[m] - deltas[m]
            for j in range(len(dfs)):
                if Q[m + 1, j + 1] != 0.0:
                    acc += self._coeff(dt * Q[m + 1, j + 1]) * dfs[j]
            out.append(acc)
        return out

    def accumulate(self, deltas):
        """Add one round of corrections to what this level owes upwards."""
        self.delta_acc = (
            deltas if self.delta_acc is None else [a + d for a, d in zip(self.delta_acc, deltas, strict=True)]
        )

    def update_nodes(self):
        r"""
        Sweep, and keep the level's bookkeeping straight if it is part of a hierarchy.

        The sweep itself is :meth:`_sweep_nodes`, which each concrete sweeper supplies. Around it:
        follow any change of :math:`u_0` that arrived after the residual was handed down, advance
        that residual by the update just applied, and bank the corrections for a transfer to prolong.

        A level that computes its own residual has no bookkeeping to do, so on a single-level run
        every line below the sweep is a no-op. That is why there is one sweeper rather than two.

        Returns
        -------
        None
        """
        self.sync_initial_value()
        self._sweep_nodes()
        if self.eps_in is None:
            return None

        deltas = [self._to_work(self.level.prob, d) for d in self._deltas]
        self.accumulate(deltas)
        self.eps_in = self.advance_residual(self.eps_in, deltas, self._dfs)
        return None

    _work_scale = 1.0
    r"""Shared divisor applied to the correction quantities before they are stored."""

    def _work_init(self, prob):
        """Build the problem's ``init`` tuple with the correction dtype substituted."""
        return (prob.init[0], prob.init[1], self._work_dtype)

    def _scales_corrections(self):
        """
        Whether this sweeper may choose a scale for its correction quantities.

        A level that computes its own residual may: everything it stores is derived from that
        residual within the same sweep, so one divisor keeps them all commensurate and ordinary
        arithmetic on them stays correct. A level that *inherits* a residual may not, because the
        inherited value was scaled by whoever produced it and is carried across sweeps.
        """
        return self.eps_in is None

    def _set_work_scale(self, values):
        r"""
        Choose the divisor for this sweep, from the residual the corrections will be built out of.

        This is what lets a correction be stored below ``float16``'s smallest normal, 6.1e-5. The
        delta form drives :math:`r` and :math:`\delta` towards zero on purpose, and half
        precision has almost no mantissa left down there -- 1.3e-2 relative at 1e-6, 1.9e-1 at 1e-7 --
        so an unscaled correction turns to noise exactly when it starts to matter. Dividing by the
        residual's own magnitude keeps the stored values at :math:`\mathcal{O}(1)`, which is block
        floating point, and is what half-precision hardware does anyway.

        Parameters
        ----------
        values : list
            The residual at the collocation nodes, in backend units.
        """
        if self._work_dtype is None or not self._scales_corrections():
            self._work_scale = 1.0
            return
        biggest = max((abs(value) for value in values), default=0.0)
        self._work_scale = float(biggest) if biggest > 0.0 else 1.0

    def _to_work(self, prob, value):
        """
        Store a small correction quantity in a reduced-precision datatype.

        Returns the value unchanged when no reduced precision was requested, so the default path
        makes no assumption about the datatype and works with any pySDC backend.

        Raises
        ------
        NotImplementedError
            If ``correction_precision`` was requested but the datatype cannot be built at another
            precision, as is the case for datatypes not backed by a numpy array.
        """
        if self._work_dtype is None:
            return value
        try:
            me = prob.dtype_u(self._work_init(prob))
            me[:] = value if self._work_scale == 1.0 else value / self._work_scale
        except (TypeError, NotImplementedError) as error:
            raise NotImplementedError(
                f'correction_precision is not supported for {prob.dtype_u.__name__}: it cannot be '
                f'built at a different precision from the problem init tuple ({error})'
            ) from error
        return me

    def _to_backend(self, prob, value):
        """
        Lift a possibly reduced-precision quantity back to backend precision.

        A no-op when no reduced precision is in play, which keeps the default path datatype-agnostic.
        """
        if self._work_dtype is None:
            return value
        me = prob.dtype_u(prob.init)
        me[:] = value
        if self._work_scale != 1.0:
            # widen first, scale second. The other order multiplies at the reduced precision, and a
            # scale of 1e-8 then lands the result below float16's smallest subnormal on the way out
            # -- the value is destroyed before it ever reaches the backend-precision array.
            me *= self._work_scale
        return me

    def _coeff(self, value):
        """
        Cast a scalar coefficient so an accumulation stays at the precision it should.

        The ``float`` matters. A coefficient taken out of a numpy array is an ``np.float64``, and
        multiplying a reduced-precision level quantity by one of those upcasts the result to
        ``float64`` under NEP 50 -- NumPy 2's rule -- which would quietly put the whole level back at
        backend precision. A plain Python float is weak under both the old and the new rule, so the
        array's own dtype wins.
        """
        if self._work_dtype is None:
            return float(value)
        return self._work_dtype.type(value)

    def _residual_nodes(self):
        r"""
        Compute :math:`r_m = u_0 + \tau_m + \Delta t (Q f^k)_m - u^k_m`.

        This is the high-precision residual of iterative refinement. It is a difference of
        :math:`\mathcal{O}(1)` quantities and is therefore always formed in backend precision --
        unless a transfer handed one down, in which case that one is already exact and small, and
        rebuilding it here is what the delta-form hierarchy exists to avoid.

        Returns
        -------
        list
            One residual per collocation node.
        """
        if self.eps_in is not None:
            return self.eps_in

        # the residual in one pass, where the sweeper's own integrate is the one in this module
        eps = _gpu_rows(self, residual=True) if type(self).integrate in _GPU_INTEGRATES else None
        if eps is not None:
            return eps
        lvl = self.level
        eps = self.integrate()
        for m in range(self.coll.num_nodes):
            eps[m] += lvl.u[0]
            eps[m] -= lvl.u[m + 1]
            if lvl.tau[m] is not None:
                eps[m] += lvl.tau[m]
        return eps

    def _f_increment(self, prob, f_new, f_old, u_old, delta, t_node):
        r"""
        Form the right-hand side increment :math:`\Delta f = f(w+\delta) - f(w)`.

        Formed by subtraction unless the problem can expand it analytically. The subtraction is a
        cancellation of two :math:`\mathcal{O}(|f|)` quantities, so its absolute error is
        :math:`\varepsilon |f|` in whatever precision the level stores ``f`` at, and :math:`|f|`
        carries the operator norm. That is harmless while the level is at backend precision and
        becomes the binding term as soon as it is not, which is why a problem living on a
        reduced-precision level should provide ``eval_f_increment``.

        Parameters
        ----------
        prob : pySDC.core.problem.Problem
            The problem on this level.
        f_new, f_old : dtype_f
            ``f`` at :math:`w+\delta` and at :math:`w`, used by the subtraction fallback.
        u_old : dtype_u
            The base state :math:`w`.
        delta : dtype_u
            The correction :math:`\delta`.
        t_node : float
            Physical time of the collocation node.

        Returns
        -------
        dtype_f
            The increment, with the same splitting as ``eval_f``.
        """
        if hasattr(prob, 'eval_f_increment'):
            increment = prob.eval_f_increment(u_old, delta, t_node)
        else:
            increment = prob.dtype_f(f_new)
            increment -= f_old
        # recorded for the residual recursion, so a transfer never recovers it by subtraction
        self._dfs.append(total_increment(prob, increment))
        return increment

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

        rhs_phys = self._to_backend(prob, rhs_corr)

        if alpha == 0:
            # explicit node: the correction is the residual itself
            delta = rhs_phys
        elif hasattr(prob, 'solve_system_delta'):
            delta = prob.solve_system_delta(rhs_phys, alpha, u_old, f_old, t_node)
        elif self._linear_implicit:
            # f(w+d) - f(w) = A d, so solve_system already solves the correction equation once the
            # affine part f(0, t) has been removed. f(0, t) vanishes for a homogeneous operator.
            affine = prob.eval_f(prob.dtype_u(prob.init, val=0.0), t_node)
            affine = affine if implicit_part is None else affine.impl
            delta = self._solve_scaled(prob, rhs_phys, alpha, affine, t_node)
        else:
            # Fallback: substitute y = u_old + delta. Always correct, but the solver sees an O(1)
            # unknown, so there is no precision benefit.
            rhs_phys += u_old
            rhs_phys -= alpha * f_impl_old
            solution = prob.solve_system(rhs_phys, alpha, u_old, t_node)
            delta = prob.dtype_u(solution)
            delta -= u_old
        # recorded so a transfer never has to recover the correction by subtraction
        self._deltas.append(delta)
        return delta

    @staticmethod
    def _solve_scaled(prob, rhs, alpha, affine, t_node):
        r"""
        Hand a linear correction solve its right-hand side at unit size, and take back any precision.

        This is the sweeper's side of the correction-solve contract documented on
        :class:`~pySDC.core.problem.Problem`. The right-hand side shrinks with the iteration, which
        is the point of the delta form and also what pushes it out of a reduced-precision format's
        range: a correction of 1e-10 is below ``float16``'s smallest subnormal. Dividing by its
        maximum before the solve and multiplying the result by it afterwards is exact for a linear
        solve, and makes an absolute solver tolerance a relative one, which the contract asks for
        anyway. The affine part :math:`\alpha f(0, t)` is removed *after* scaling: the stock solve
        adds it back in unscaled, and the two only cancel at the same scale. The result may come back in any precision; it is widened before it is scaled back,
        since scaling first would multiply at the solver's precision.

        Parameters
        ----------
        prob : pySDC.core.problem.Problem
            The level's problem.
        rhs : dtype_u
            Right-hand side of the correction equation, modified in place.
        alpha : float
            Implicit prefactor.
        affine : dtype_u
            The implicit right-hand side at zero, :math:`f(0, t)`, which a homogeneous operator
            makes zero.
        t_node : float
            Physical time of the collocation node.

        Returns
        -------
        dtype_u
            The correction, at the level's precision.
        """
        zero = prob.dtype_u(prob.init, val=0.0)
        scale = float(abs(rhs))
        if scale == 0.0:
            return zero
        rhs /= scale
        rhs -= alpha * affine
        solution = prob.solve_system(rhs, alpha, zero, t_node)
        if isinstance(prob.init, tuple) and getattr(solution, 'dtype', None) not in (None, np.dtype(prob.init[-1])):
            delta = prob.dtype_u(prob.init)
            delta[:] = solution
        else:
            delta = prob.dtype_u(solution)
        delta *= scale
        return delta


class delta_implicit(DeltaFormMixin, generic_implicit):
    """
    Delta-form counterpart of :class:`~pySDC.implementations.sweeper_classes.generic_implicit.generic_implicit`.

    Mathematically identical to the standard sweep; see the module docstring for the sweeper
    parameters ``correction_precision`` and ``linear_implicit``.
    """

    def integrate(self):
        """
        Integrate the right-hand side, as the parent sweeper does; on a GPU in one kernel per node,
        with bitwise the same result.

        Returns
        -------
        list
            :math:`\\Delta t\\, (Q F)_m` at every node.
        """
        rows = _gpu_rows(self, residual=False)
        return super().integrate() if rows is None else rows

    def _sweep_nodes(self):
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

        residual = self._residual_nodes()
        self._set_work_scale(residual)
        eps = [self._to_work(prob, value) for value in residual]
        df = [None] * (num_nodes + 1)

        for m in range(num_nodes):
            t_node = lvl.time + lvl.dt * self.coll.nodes[m]

            rhs_corr = type(eps[m])(eps[m])
            for j in range(1, m + 1):
                if self.QI[m + 1, j] != 0.0:
                    rhs_corr += self._coeff(lvl.dt * self.QI[m + 1, j]) * df[j]

            alpha = lvl.dt * self.QI[m + 1, m + 1]
            # no copies: both are read only (the correction-solve contract), and the level gets new objects below
            u_old = lvl.u[m + 1]
            f_old = lvl.f[m + 1]

            delta = self._solve_correction(rhs_corr, alpha, u_old, f_old, t_node)

            lvl.u[m + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
            lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], t_node)

            increment = self._f_increment(prob, lvl.f[m + 1], f_old, u_old, delta, t_node)
            df[m + 1] = self._to_work(prob, increment)

        lvl.status.updated = True
        return None


class delta_imex_1st_order(DeltaFormMixin, imex_1st_order):
    """
    Delta-form counterpart of :class:`~pySDC.implementations.sweeper_classes.imex_1st_order.imex_1st_order`.

    The correction equation contains only differences of ``f``, never a Jacobian, so the
    explicit/implicit splitting is untouched.
    """

    def integrate(self):
        """
        Integrate the right-hand side, as the parent sweeper does; on a GPU in one kernel per node,
        with bitwise the same result.

        Returns
        -------
        list
            :math:`\\Delta t\\, (Q F)_m` at every node.
        """
        rows = _gpu_rows(self, residual=False)
        return super().integrate() if rows is None else rows

    def _sweep_nodes(self):
        """
        Perform one delta-form IMEX sweep over all collocation nodes.

        ``QE`` is strictly lower triangular, which
        :class:`~pySDC.implementations.sweeper_classes.imex_1st_order.imex_1st_order` already enforces, so the
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

        residual = self._residual_nodes()
        self._set_work_scale(residual)
        eps = [self._to_work(prob, value) for value in residual]
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
            # no copies: both are read only (the correction-solve contract), and the level gets new objects below
            u_old = lvl.u[m + 1]
            f_old = lvl.f[m + 1]

            delta = self._solve_correction(rhs_corr, alpha, u_old, f_old, t_node, implicit_part=f_old.impl)

            lvl.u[m + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
            lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], t_node)

            increment = self._f_increment(prob, lvl.f[m + 1], f_old, u_old, delta, t_node)
            df_impl[m + 1] = self._to_work(prob, prob.dtype_u(increment.impl))
            df_expl[m + 1] = self._to_work(prob, prob.dtype_u(increment.expl))

        lvl.status.updated = True
        return None


_GPU_INTEGRATES = (delta_implicit.integrate, delta_imex_1st_order.integrate)


class DeltaFormMassMixin(DeltaFormMixin):
    r"""
    The delta form for problems in mass-matrix form, :math:`M u' = F(u)`, as the mass-matrix sweepers solve them.

    The collocation residual is then a load vector, :math:`r_m = M u_0 + \tau_m + \Delta t (Q F^k)_m - M u^k_m`, and
    the correction equation at node :math:`m` becomes

    .. math::
        M \delta_m - \alpha \left[F(u^k_m + \delta_m) - F(u^k_m)\right]
        = r_m + \Delta t \sum_{j<m} Q^\Delta_{mj}\, \Delta F_j.

    Everything else is the delta form of :class:`DeltaFormMixin`. As with the mass-matrix sweepers, the initial value
    below the finest level is already the load vector :math:`M u_0`.

    The correction is solved by the problem's ``solve_system_delta``, which has to solve the equation above, or by
    substituting :math:`y = u^k_m + \delta_m` into ``solve_system``, which is exact. The latter also handles a node with
    :math:`\alpha = 0`, where the correction needs a mass solve. ``linear_implicit`` is not available: it relies on
    ``solve_system`` for the correction itself, which would impose the boundary data on the correction instead of
    the difference between the data and :math:`u^k_m`.
    """

    def _delta_setup(self):
        """Read the optional sweeper parameters and refuse those the mass form cannot honour."""
        super()._delta_setup()
        if self._linear_implicit:
            raise ParameterError(
                'linear_implicit is not available for the mass-matrix delta form: solve_system would impose the '
                'boundary data on the correction, which needs the difference between the data and the current value'
            )

    def _residual_nodes(self):
        r"""
        Compute the residual :math:`r_m = M u_0 + \tau_m + \Delta t (Q F^k)_m - M u^k_m`, a load vector, unless a
        transfer handed one down.

        Returns
        -------
        list
            One residual per collocation node.
        """
        if self.eps_in is not None:
            return self.eps_in
        lvl = self.level
        prob = lvl.prob
        eps = self.integrate()
        for m in range(self.coll.num_nodes):
            # below the finest level, u0 is already the load vector M u0
            if lvl.level_index == 0:
                eps[m] += prob.apply_mass_matrix(lvl.u[0] - lvl.u[m + 1])
            else:
                eps[m] += lvl.u[0] - prob.apply_mass_matrix(lvl.u[m + 1])
            if lvl.tau[m] is not None:
                eps[m] += lvl.tau[m]
        return eps

    def advance_residual(self, eps, deltas, dfs):
        r"""
        Advance a residual by an update, :math:`r \leftarrow r - M \delta + \Delta t (Q \Delta F)`.

        Parameters
        ----------
        eps, deltas, dfs : list
            As for :meth:`DeltaFormMixin.advance_residual`; the corrections are applied to the nodal values, so the
            residual changes by their load vectors.

        Returns
        -------
        list
            The advanced residual.
        """
        prob = self.level.prob
        return super().advance_residual(eps, [prob.apply_mass_matrix(delta) for delta in deltas], dfs)

    def _solve_correction(self, rhs_corr, alpha, u_old, f_old, t_node, implicit_part=None):
        """
        Solve the node-local correction equation of the mass form, see the class docstring.

        Parameters
        ----------
        rhs_corr : dtype_u
            Right-hand side of the correction equation, a load vector.
        alpha : float
            Implicit prefactor.
        u_old : dtype_u
            Current nodal value, the base state of the correction.
        f_old : dtype_f
            ``F`` at ``u_old``.
        t_node : float
            Physical time of the collocation node.
        implicit_part : dtype_u, optional
            The implicit component of ``f_old`` for IMEX problems. Defaults to ``f_old``.

        Returns
        -------
        dtype_u
            The correction.
        """
        prob = self.level.prob
        rhs = self._to_backend(prob, rhs_corr)
        if hasattr(prob, 'solve_system_delta'):
            delta = prob.solve_system_delta(rhs, alpha, u_old, f_old, t_node)
        else:
            # substitute y = u_old + delta: M y - alpha F_I(y) = rhs + M u_old - alpha F_I(u_old)
            f_impl_old = f_old if implicit_part is None else implicit_part
            rhs = rhs + prob.apply_mass_matrix(u_old) - alpha * f_impl_old
            delta = prob.dtype_u(prob.solve_system(rhs, alpha, u_old, t_node))
            delta -= u_old
        self._deltas.append(delta)
        return delta


class delta_implicit_mass(DeltaFormMassMixin, generic_implicit_mass):
    """
    Delta-form counterpart of :class:`~pySDC.implementations.sweeper_classes.generic_implicit_mass.generic_implicit_mass`,
    for problems in mass-matrix form. Mathematically identical to it; see :class:`DeltaFormMassMixin`.
    """

    _sweep_nodes = delta_implicit._sweep_nodes


class delta_imex_1st_order_mass(DeltaFormMassMixin, imex_1st_order_mass):
    """
    Delta-form counterpart of :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`,
    for problems in mass-matrix form. Mathematically identical to it; see :class:`DeltaFormMassMixin`.
    """

    _sweep_nodes = delta_imex_1st_order._sweep_nodes


def total_increment(prob, increment):
    """
    The full right-hand side increment, recombining an IMEX splitting.

    Parameters
    ----------
    prob : pySDC.core.problem.Problem
        The problem the increment belongs to.
    increment : dtype_f
        The increment, possibly split into ``impl`` and ``expl``.

    Returns
    -------
    dtype_u
        The sum of the parts.
    """
    if not hasattr(increment, 'impl'):
        return increment
    return prob.dtype_u(increment.impl) + increment.expl
