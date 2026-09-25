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
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

TOLERANCE_SAFETY = 100.0
"""Unconditional minimum multiple of the working-precision epsilon."""

CONDITIONING_SAFETY = 4.0
"""Multiplier applied to the conditioning estimate ``1 + alpha * ||J||``."""


class allencahn_delta(allencahn_fullyimplicit):
    r"""
        Fully implicit Allen-Cahn with a node-local correction solve.

        The right-hand side is :math:`f(u) = Au + \frac{1}{2}\varepsilon^{-2} v (1 - v^\nu)` with
        :math:`v = 2u - 1`, wells at 0 and 1, so for :math:`\nu=2`, writing :math:`d = 2\delta`,

        .. math::
            f(w+\delta) - f(w) = A\delta
                + \tfrac{1}{2}\varepsilon^{-2}\left[d - \left(3v^2 d + 3v d^2 + d^3\right)\right],

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

    def __init__(self, solve_precision=None, normalize=True, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        if self.nu != 2:
            raise NotImplementedError('allencahn_delta derives its analytic increment for nu=2 only!')

        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        self.normalize = normalize
        dtype = np.dtype('float64') if self.solve_precision is None else self.solve_precision
        # What the solve *stores* is ``solve_precision``; what it computes in is at least float32,
        # because SciPy holds no float16 sparse matrix and fp16 hardware accumulates in fp32 anyway.
        # Same convention the genuinely reduced-precision levels use for their operators.
        compute = np.promote_types(dtype, np.float32)
        size = self.nvars[0] * self.nvars[1]
        self._work_dtype = dtype
        self._compute_dtype = compute
        self._A_work = self.A.astype(compute).tocsr()
        self._Id_work = sp.eye(size, dtype=compute, format='csr')
        self._inv_eps2 = compute.type(1.0 / self.eps**2)

        # Bound on ||J||_inf, used to make the tolerance floor conditioning-aware. The reaction
        # term contributes |1 - (nu+1) v^nu| / eps^2 <= nu / eps^2 for v = 2u - 1 in [-1, 1].
        self._operator_norm = float(abs(self.A).sum(axis=1).max()) + self.nu / self.eps**2

    def _increment(self, base, delta, matrix=None, inv_eps2=None, scale=1.0):
        r"""
        Evaluate :math:`s^{-1}[f(w+s\delta) - f(w)]` without cancellation.

        Parameters
        ----------
        base : numpy.ndarray
            Flattened base state :math:`w` at working precision.
        delta : numpy.ndarray
            Flattened correction :math:`\delta` at working precision.
        matrix : scipy.sparse.spmatrix, optional
            Diffusion operator to use, defaulting to the one at ``solve_precision``. The sweeper's
            own increment wants the backend-precision operator instead, which is the only difference
            between the two callers -- the expansion is the same and lives here once.
        inv_eps2 : float, optional
            :math:`\varepsilon^{-2}`, likewise.
        scale : float, optional
            Normalisation :math:`s` applied to the unknown, so ``delta`` is
            :math:`\delta/s`. The expansion stays exact -- the quadratic and cubic terms simply
            pick up :math:`s` and :math:`s^2` -- which is what lets the whole solve run on an
            :math:`\mathcal{O}(1)` unknown. Defaults to 1, the unnormalised increment.

        Returns
        -------
        numpy.ndarray
            The increment, divided by ``scale``.
        """
        matrix = self._A_work if matrix is None else matrix
        inv_eps2 = self._inv_eps2 if inv_eps2 is None else inv_eps2
        # The reaction is (1 / 2 eps^2) v (1 - v^2) in v = 2u - 1 (wells at 0 and 1), so expand in
        # v, where an increment delta in u is 2 delta.
        v, d = 2.0 * base - 1.0, 2.0 * delta
        cubic = 3.0 * v * v * d + 3.0 * scale * v * d * d + scale**2 * d**3
        return matrix.dot(delta) + 0.5 * inv_eps2 * (d - cubic)

    def eval_f_increment(self, base, delta, t):
        r"""
        Evaluate :math:`f(w+\delta) - f(w)` at backend precision, without cancellation.

        Same expansion as :meth:`_increment`, which the node-local solve uses at its own working
        precision. This one is for the sweeper, whose accumulation lives on the level, so it is
        formed at the level's precision instead.

        Parameters
        ----------
        base : dtype_u
            The base state :math:`w`.
        delta : dtype_u
            The correction :math:`\delta`.
        t : float
            Physical time, accepted for interface compatibility.

        Returns
        -------
        dtype_f
            The increment.
        """
        w = np.asarray(base, dtype=float).reshape(-1)
        d = np.asarray(delta, dtype=float).reshape(-1)
        me = self.dtype_f(self.init)
        me[:] = self._increment(w, d, matrix=self.A, inv_eps2=1.0 / self.eps**2).reshape(self.nvars)
        return me

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
        dtype = self._compute_dtype
        # the derivative of the reaction, as `reaction_prime` has it
        v = dtype.type(2.0) * state - dtype.type(1.0)
        diagonal = (dtype.type(1.0) - dtype.type(self.nu + 1) * v**self.nu).astype(dtype)
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
        store, dtype = self._work_dtype, self._compute_dtype
        alpha = dtype.type(factor)
        base_work = np.asarray(base, dtype=dtype).reshape(-1)
        rhs64 = np.asarray(r, dtype=np.float64).reshape(-1)
        rhs_scale = max(float(np.linalg.norm(rhs64, np.inf)), 1e-300)

        # Normalise the unknown to O(1). The delta form hands the solver a correction that shrinks
        # with the sweeps, and float16's smallest normal is 6.1e-5 -- so without this the unknown
        # itself underflows to zero after a handful of iterations and the solve returns noise.
        # Exact here as well as in the linear case, because the increment carries the scale.
        scale = rhs_scale if self.normalize else 1.0
        rhs_work = np.asarray((rhs64 / scale).astype(store), dtype=dtype)
        delta = np.zeros_like(rhs_work)

        # Raise the inherited tolerances to what this working precision can actually deliver.
        # Asking for less than that does not fail loudly, it just runs to newton_maxiter against an
        # impossible bar: removing this floor costs 20x the linear work in float32 for an identical
        # answer. The floor scales with alpha*||J|| because forming the correction residual involves
        # alpha*J*delta, so its rounding error is O(eps * alpha*||J|| * |delta|); alpha is only
        # known here, which is why this cannot be decided in the frontend.
        conditioning = max(TOLERANCE_SAFETY, CONDITIONING_SAFETY * (1.0 + float(factor) * self._operator_norm))
        # The Krylov solve runs at the compute type, the Newton bar is on a correction *stored* at
        # the reduced one -- so the two floors come from different epsilons when the pair differs.
        krylov_tol = max(float(self.lin_tol), np.finfo(dtype).eps * conditioning)
        bar = max(float(self.newton_tol), np.finfo(store).eps * conditioning * rhs_scale)

        converged = False
        for _ in range(self.newton_maxiter):
            residual = delta - alpha * self._increment(base_work, delta, scale=scale) - rhs_work
            if float(np.linalg.norm(residual, np.inf)) * scale < bar:
                converged = True
                break
            step = cg(
                self._jacobian(base_work + scale * delta, alpha),
                residual,
                x0=np.zeros_like(residual),
                rtol=krylov_tol,
                maxiter=self.lin_maxiter,
                atol=0,
                callback=self.work_counters['linear'],
            )[0]
            delta = np.asarray((delta - step).astype(store), dtype=dtype)
            self.work_counters['newton']()

        if not converged:
            self.logger.warning(
                'Correction solve hit newton_maxiter=%d at %s without reaching a residual of %.2e. '
                'The tolerance is probably below what this working precision can deliver; raise '
                'newton_tol.',
                self.newton_maxiter,
                store,
                bar,
            )

        me = self.dtype_u(self.init)
        me[:] = (scale * delta.astype(np.float64)).reshape(self.nvars)
        return me


class heat_delta(heatNd_unforced):
    r"""
    Heat equation with an analytic increment and a reduced-precision node-local solve.

    The implicit operator is linear, so no ``solve_system_delta`` is needed: the sweeper reaches the
    correction equation with ``linear_implicit=True`` and the stock ``solve_system``. What this class
    adds is the two things a *level* below backend precision needs.

    ``eval_f_increment`` is inherited: the operator is linear, so
    :class:`GenericNDimFinDiff` supplies :math:`A\delta` for every problem built on it.

    ``solve_precision``
        Emulated, as on the PETSc and FEniCS backends: the operator, the right-hand side and the
        result are rounded through the working precision while the arithmetic stays at the backend
        type. SciPy carries no ``float16`` sparse matrix, so half precision can only be reached this
        way.

    ``normalize``
        Scales the right-hand side to :math:`\mathcal{O}(1)` before the solve and scales the result
        back, which is exact for a linear solve. Half precision needs it: the smallest ``float16``
        subnormal is 6e-8, so a correction of 1e-10 -- exactly what the delta form is built to hand
        the solver -- rounds to **zero** without it. The two are otherwise in direct tension.

    Parameters
    ----------
    solve_precision : dtype-like or None, optional
        Working precision to emulate for the node-local solve. ``None`` keeps backend precision.
    normalize : bool, optional
        Scale the right-hand side to :math:`\mathcal{O}(1)` around the solve.
    **kwargs
        Forwarded to :class:`heatNd_unforced`.
    """

    def __init__(self, solve_precision=None, normalize=True, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        self.normalize = normalize

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - factor\,A)\,x = rhs`, at an emulated reduced precision if asked.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess, unused for this direct solve.
        t : float
            Current time.

        Returns
        -------
        dtype_u
            The solution.
        """
        if self.solve_precision is None:
            return super().solve_system(rhs, factor, u0, t)

        dtype = self.solve_precision
        b = np.asarray(rhs, dtype=np.float64).flatten()
        scale = max(float(np.max(np.abs(b))), 1e-300) if self.normalize else 1.0
        b = np.asarray((b / scale).astype(dtype), dtype=np.float64)
        matrix = np.asarray((self.Id - factor * self.A).todense().astype(dtype), dtype=np.float64)
        solution = np.linalg.solve(matrix, b)
        me = self.dtype_u(self.init)
        me[:] = (scale * np.asarray(solution.astype(dtype), dtype=np.float64)).reshape(self.nvars)
        return me


class heat_no_increment(heat_delta):
    r"""
    Control: the heat equation with its analytic increment deliberately out of reach.

    Every problem built on :class:`GenericNDimFinDiff` supplies ``eval_f_increment``, so a sweeper on
    one never falls back to forming :math:`\Delta f` by subtracting two stored right-hand sides.
    This class hides it again, which is what makes the claim that the increment matters falsifiable.

    The attribute raises rather than being absent, because the sweeper dispatches on ``hasattr`` and
    a raising property is the way to make that report ``False`` for an inherited method.
    """

    @property
    def eval_f_increment(self):
        """
        Raises
        ------
        AttributeError
            Always. That is the point of the class.
        """
        raise AttributeError('control: the analytic increment is deliberately unavailable here')
