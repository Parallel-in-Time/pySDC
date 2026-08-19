r"""
PETSc problem with a node-local correction solve.

Kept in its own module because importing it requires ``petsc4py``, which :mod:`problems` does not.

The stock PETSc problem defines a residual callback ``formFunction(snes, X, F): F = X - factor
f(X)`` and lets SNES solve ``F(X) = rhs``. The delta variant is the *same callback shape* with the
correction as the unknown, and ``formJacobian`` is reused verbatim, simply evaluated at
:math:`w + \delta`. That is the whole change: nothing about the solver setup differs.

For Fisher, :math:`f(u) = u_{xx} + \lambda_0^2 u (1 - u^\nu)`, so the increment is

.. math::
    f(w+\delta) - f(w) = \delta_{xx}
        + \lambda_0^2\left(\delta - \left[(w+\delta)^{\nu+1} - w^{\nu+1}\right]\right),

and the bracket is expanded binomially so every term carries an explicit factor :math:`\delta`,
which keeps the increment free of cancellation.

Reduced precision is **emulated**: PETSc's scalar type is fixed at build time, so a genuine
single-precision KSP needs a separate ``--with-precision=single`` build. Values are instead rounded
through the requested working precision and stored back as PETSc scalars, which caps the
*information* at that precision while the arithmetic stays at the backend scalar type. This is
optimistic about both iteration counts and attainable accuracy compared with a real
single-precision build, and results obtained this way should be labelled as emulated.
"""

from math import comb

import numpy as np
from petsc4py import PETSc

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import (
    Fisher_full,
    petsc_fisher_fullyimplicit,
)

TOLERANCE_SAFETY = 100.0
"""Unconditional minimum multiple of the working-precision epsilon."""


def quantize(values, work_precision):
    """
    Round an array through the working precision and back to the PETSc scalar type.

    Parameters
    ----------
    values : array-like
        Values to quantize.
    work_precision : numpy.dtype or None
        Working precision. ``None`` leaves the values untouched.

    Returns
    -------
    numpy.ndarray
        The quantized values, in PETSc's scalar type.
    """
    if work_precision is None:
        return np.asarray(values)
    return np.asarray(np.asarray(values, dtype=np.dtype(work_precision)), dtype=PETSc.ScalarType)


def quantize_scalar(value, work_precision):
    """
    Round a scalar through the working precision.

    Parameters
    ----------
    value : float
        Value to quantize.
    work_precision : numpy.dtype or None
        Working precision. ``None`` leaves the value untouched.

    Returns
    -------
    float
        The quantized value.
    """
    if work_precision is None:
        return float(value)
    return float(np.dtype(work_precision).type(value))


def binomial_increment(base, delta, power):
    r"""
    Expand :math:`(w+\delta)^{p} - w^{p}` so every term carries a factor :math:`\delta`.

    Parameters
    ----------
    base : float
        The base state :math:`w`.
    delta : float
        The correction :math:`\delta`.
    power : int
        The exponent :math:`p`.

    Returns
    -------
    float
        The increment, free of cancellation.
    """
    out = 0.0
    for k in range(1, power + 1):
        out += comb(power, k) * base ** (power - k) * delta**k
    return out


class FisherCorrectionResidual(Fisher_full):
    """Residual callback for the correction problem. Same shape as the stock one."""

    base_vec = None
    work_precision = None

    def set_context(self, base_vec, work_precision):
        """
        Supply the base state and the working precision for this solve.

        Parameters
        ----------
        base_vec : PETSc.Vec
            The base state :math:`w`.
        work_precision : numpy.dtype or None
            Working precision to emulate.
        """
        self.base_vec = base_vec
        self.work_precision = work_precision

    def formFunction(self, snes, X, F):
        r"""
        Evaluate :math:`\delta - factor\,[f(w+\delta) - f(w)]`.

        Parameters
        ----------
        snes : PETSc.SNES
            The nonlinear solver.
        X : PETSc.Vec
            The current correction.
        F : PETSc.Vec
            Output vector, overwritten.

        Returns
        -------
        None
        """
        self.da.globalToLocal(X, self.localX)
        delta = self.da.getVecArray(self.localX)
        out = self.da.getVecArray(F)

        local_base = self.da.createLocalVec()
        self.da.globalToLocal(self.base_vec, local_base)
        base = self.da.getVecArray(local_base)

        lam2 = self.prob.lambda0**2
        nu = self.prob.nu
        wp = self.work_precision

        for i in range(self.xs, self.xe):
            if i == 0 or i == self.mx - 1:
                out[i] = quantize_scalar(delta[i], wp)
            else:
                d_here = quantize_scalar(delta[i], wp)
                d_east = quantize_scalar(delta[i + 1], wp)
                d_west = quantize_scalar(delta[i - 1], wp)
                w_here = quantize_scalar(base[i], wp)
                d_xx = quantize_scalar((d_east - 2 * d_here + d_west) / self.dx**2, wp)
                reaction = quantize_scalar(lam2 * (d_here - binomial_increment(w_here, d_here, nu + 1)), wp)
                out[i] = quantize_scalar(d_here - self.factor * (d_xx + reaction), wp)

    def formJacobian(self, snes, X, J, P):
        """
        Reuse the stock Jacobian, evaluated at ``base + X``.

        Parameters
        ----------
        snes : PETSc.SNES
            The nonlinear solver.
        X : PETSc.Vec
            The current correction.
        J : PETSc.Mat
            Jacobian matrix.
        P : PETSc.Mat
            Preconditioner matrix.

        Returns
        -------
        The return value of the stock ``formJacobian``.
        """
        shifted = X.duplicate()
        X.copy(shifted)
        shifted.axpy(1.0, self.base_vec)
        return super().formJacobian(snes, shifted, J, P)


class petsc_fisher_delta(petsc_fisher_fullyimplicit):
    """
    PETSc Fisher exposing ``solve_system_delta`` alongside the stock ``solve_system``.

    Parameters
    ----------
    solve_precision : dtype-like or None, optional
        Working precision to emulate for the node-local solve. ``None`` keeps backend precision.
    **kwargs
        Forwarded to :class:`petsc_fisher_fullyimplicit`.
    """

    def __init__(self, solve_precision=None, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)

    def solve_system_delta(self, r, factor, base, f_base, t):
        r"""
        Solve :math:`\delta - factor\,[f(base+\delta) - f(base)] = r` for the correction.

        Parameters
        ----------
        r : dtype_u
            Right-hand side of the correction equation.
        factor : float
            Implicit prefactor assembled by the sweeper.
        base : dtype_u
            Base state :math:`w`.
        f_base : dtype_f
            ``f`` evaluated at ``base``; accepted so no extra evaluation is needed.
        t : float
            Physical time, accepted for interface compatibility.

        Returns
        -------
        dtype_u
            The correction.
        """
        target = FisherCorrectionResidual(da=self.init, prob=self, factor=factor, dx=self.dx)
        target.set_context(base, self.solve_precision)

        # the KSP cannot beat the emulated precision, so do not ask it to
        if self.solve_precision is not None:
            floor = TOLERANCE_SAFETY * float(np.finfo(self.solve_precision).eps)
            self.snes.getKSP().setTolerances(rtol=max(self.lsol_tol, floor))

        rhs = self.dtype_u(self.init)
        self.init.getVecArray(rhs)[...] = quantize(self.init.getVecArray(r)[...], self.solve_precision)

        residual_vec = self.init.createGlobalVec()
        self.snes.setFunction(target.formFunction, residual_vec)
        jacobian = self.init.createMatrix()
        self.snes.setJacobian(target.formJacobian, jacobian)

        me = self.dtype_u(self.init)
        me.set(0.0)
        self.snes.solve(rhs, me)
        self.init.getVecArray(me)[...] = quantize(self.init.getVecArray(me)[...], self.solve_precision)

        self.snes_itercount += self.snes.getIterationNumber()
        self.snes_ncalls += 1
        return me
