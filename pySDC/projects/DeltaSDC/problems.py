r"""
Problem classes exposing a node-local *correction* solve.

Only problems with a **nonlinear** implicit operator need this. For a linear or affine implicit
operator the stock ``solve_system`` already solves the correction equation, and the delta-form
sweeper reaches it with ``linear_implicit=True`` and no problem-class change at all.

The contract is

.. math::
    \texttt{solve\_system\_delta}(r, \alpha, w, f_w, t) \;\rightarrow\; \delta
    \quad\text{solving}\quad
    \delta - \alpha\,[f(w+\delta) - f(w)] = r.

Two properties matter and both are load-bearing:

1. The **unknown is a correction**. Its magnitude falls with the sweeps, so a reduced-precision
   solve introduces an error proportional to :math:`|\delta|` rather than to :math:`|u|`.
2. The **increment must be free of cancellation**. Evaluating :math:`f(w+\delta) - f(w)` as a
   difference of two :math:`\mathcal{O}(|f|)` quantities reinstates an absolute error of order
   :math:`\varepsilon |f|`, which destroys the first property. The increment is therefore expanded
   analytically so that every term carries an explicit factor :math:`\delta`.

``f_w`` is passed in because the sweeper already holds it on the level, so honouring the contract
costs no extra right-hand side evaluation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit

TOLERANCE_SAFETY = 100.0
"""Unconditional minimum multiple of the working-precision epsilon."""

CONDITIONING_SAFETY = 4.0
"""Multiplier applied to the conditioning estimate ``1 + alpha * ||J||``."""


class allencahn_delta(allencahn_fullyimplicit):
    r"""
        Fully implicit Allen-Cahn with a node-local correction solve.

        The right-hand side is :math:`f(u) = Au + \varepsilon^{-2} u (1 - u^\nu)`, so for :math:`\nu=2`

        .. math::
            f(w+\delta) - f(w) = A\delta
                + \varepsilon^{-2}\left[\delta - \left(3w^2\delta + 3w\delta^2 + \delta^3\right)\right],

        in which every term carries a factor :math:`\delta`.

    Tolerances are the inherited ones and are set in the frontend as usual: ``lin_tol`` is the
        relative Krylov tolerance and ``newton_tol`` the absolute bar on the correction residual. Both
        keep exactly the meaning they have in :class:`allencahn_fullyimplicit`, because the correction
        residual :math:`\delta - \alpha[f(w+\delta) - f(w)] - r` is the same quantity as the parent's
        :math:`u - \alpha f(u) - rhs` under :math:`u = w + \delta`. The only addition is that both are
        raised to what ``solve_precision`` can actually deliver, which can only be decided inside the
        solve because it depends on :math:`\alpha`.

        Parameters
        ----------
        solve_precision : dtype-like or None, optional
            Working precision of the node-local solve. ``None`` keeps backend precision.
        **kwargs
            Forwarded to :class:`allencahn_fullyimplicit`, including ``lin_tol``, ``newton_tol``,
            ``lin_maxiter`` and ``newton_maxiter``.

        Raises
        ------
        NotImplementedError
            If ``nu`` is not 2, for which the analytic increment below is derived.
    """

    def __init__(self, solve_precision=None, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        if self.nu != 2:
            raise NotImplementedError('allencahn_delta derives its analytic increment for nu=2 only!')

        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        dtype = np.dtype('float64') if self.solve_precision is None else self.solve_precision
        size = self.nvars[0] * self.nvars[1]
        self._work_dtype = dtype
        self._A_work = self.A.astype(dtype).tocsr()
        self._Id_work = sp.eye(size, dtype=dtype, format='csr')
        self._inv_eps2 = dtype.type(1.0 / self.eps**2)

        # Bound on ||J||_inf, used to make the tolerance floor conditioning-aware. The reaction
        # term contributes |1 - (nu+1) u^nu| <= nu for u in [-1, 1].
        self._operator_norm = float(abs(self.A).sum(axis=1).max()) + self.nu / self.eps**2

    def _increment(self, base, delta):
        r"""
        Evaluate :math:`f(w+\delta) - f(w)` without cancellation.

        Parameters
        ----------
        base : numpy.ndarray
            Flattened base state :math:`w` at working precision.
        delta : numpy.ndarray
            Flattened correction :math:`\delta` at working precision.

        Returns
        -------
        numpy.ndarray
            The increment, at working precision.
        """
        dtype = self._work_dtype
        cubic = dtype.type(3.0) * base * base * delta + dtype.type(3.0) * base * delta * delta + delta**3
        return self._A_work.dot(delta) + self._inv_eps2 * (delta - cubic)

    def _jacobian(self, state, alpha):
        r"""
        Assemble :math:`I - \alpha J(u)` directly at working precision.

        Parameters
        ----------
        state : numpy.ndarray
            Flattened state at which the Jacobian is evaluated.
        alpha : numpy.dtype
            Implicit prefactor, already cast to working precision.

        Returns
        -------
        scipy.sparse.csr_matrix
            The system matrix, at working precision.
        """
        dtype = self._work_dtype
        diagonal = (dtype.type(1.0) - dtype.type(self.nu + 1) * state**self.nu).astype(dtype)
        jacobian = self._A_work + self._inv_eps2 * sp.diags(diagonal, offsets=0, format='csr')
        return (self._Id_work - alpha * jacobian).astype(dtype).tocsr()

    def solve_system_delta(self, r, factor, base, f_base, t):
        r"""
        Solve :math:`\delta - factor\,[f(base+\delta) - f(base)] = r` for the correction.

        The whole solve runs at ``solve_precision``: the operator, the right-hand side, the Krylov
        iterations and the returned correction. The sweeper accumulates the result into the
        backend-precision nodal value.

        Parameters
        ----------
        r : dtype_u
            Right-hand side of the correction equation.
        factor : float
            Implicit prefactor assembled by the sweeper.
        base : dtype_u
            Base state :math:`w` around which the correction is taken.
        f_base : dtype_f
            ``f`` evaluated at ``base``; accepted so no extra evaluation is needed.
        t : float
            Physical time, accepted for interface compatibility.

        Returns
        -------
        dtype_u
            The correction, in backend precision.
        """
        dtype = self._work_dtype
        alpha = dtype.type(factor)
        base_work = np.asarray(base, dtype=dtype).reshape(-1)
        rhs_work = np.asarray(r, dtype=dtype).reshape(-1)
        delta = np.zeros_like(rhs_work)

        # Raise the inherited tolerances to what this working precision can actually deliver.
        # Asking for less than that does not fail loudly, it just runs to newton_maxiter against an
        # impossible bar: removing this floor costs 20x the linear work in float32 for an identical
        # answer. The floor scales with alpha*||J|| because forming the correction residual involves
        # alpha*J*delta, so its rounding error is O(eps * alpha*||J|| * |delta|); alpha is only
        # known here, which is why this cannot be decided in the frontend.
        floor = np.finfo(dtype).eps * max(
            TOLERANCE_SAFETY, CONDITIONING_SAFETY * (1.0 + float(factor) * self._operator_norm)
        )
        krylov_tol = max(float(self.lin_tol), floor)
        scale = max(float(np.linalg.norm(rhs_work, np.inf)), 1e-300)
        bar = max(float(self.newton_tol), floor * scale)

        converged = False
        for _ in range(self.newton_maxiter):
            residual = delta - alpha * self._increment(base_work, delta) - rhs_work
            if float(np.linalg.norm(residual, np.inf)) < bar:
                converged = True
                break
            step = cg(
                self._jacobian(base_work + delta, alpha),
                residual,
                x0=np.zeros_like(residual),
                rtol=krylov_tol,
                maxiter=self.lin_maxiter,
                atol=0,
                callback=self.work_counters['linear'],
            )[0]
            delta = delta - step.astype(dtype)
            self.work_counters['newton']()

        if not converged:
            self.logger.warning(
                'Correction solve hit newton_maxiter=%d at %s without reaching a residual of %.2e. '
                'The tolerance is probably below what this working precision can deliver; raise '
                'newton_tol.',
                self.newton_maxiter,
                dtype,
                bar,
            )

        me = self.dtype_u(self.init)
        me[:] = delta.reshape(self.nvars)
        return me
