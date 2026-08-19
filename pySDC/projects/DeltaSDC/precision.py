r"""
Precision-aware tolerance policy.

A solver working at precision :math:`\varepsilon` cannot drive a relative residual much below
:math:`\mathcal{O}(\kappa\varepsilon)`. Asking it to do so does not fail loudly: the solver simply
runs to ``maxiter`` against a bar it can never meet, which turns reduced precision into a large
pessimisation instead of a saving.

This module derives the tolerance floor from the working precision instead of inheriting a
precision-agnostic constant::

    tol_floor(dtype, conditioning) = eps(dtype) * max(safety, conditioning_safety * conditioning)

The floor has to account for the *conditioning* of the node-local system, not only for the machine
epsilon. Forming the correction residual :math:`g = \delta - \alpha[f(w+\delta) - f(w)] - r`
involves the term :math:`\alpha J \delta`, so its rounding error is of order
:math:`\varepsilon\,\alpha\|J\|\,|\delta|` while the bar is :math:`rtol\,|r|`. Since
:math:`|\delta| \sim |r|`, the smallest reachable relative tolerance scales with
:math:`\alpha\|J\|`, which grows with the step size and with the spatial resolution. A fixed
multiple of ``eps`` is therefore too tight for stiff or well-resolved problems, and the solver
silently runs to ``maxiter`` instead of converging.

Callers that know :math:`\alpha\|J\|` should pass ``conditioning=1 + alpha * norm``; ``safety``
remains an unconditional minimum for callers that do not. Requested tolerances tighter than the
floor are clamped, and the clamping is recorded so it is never silent.

==========  =========  ==================  ==================
dtype       eps        floor (cond. 1)     floor (cond. 200)
==========  =========  ==================  ==================
``float64``  2.22e-16   2.22e-14            1.78e-13
``float32``  1.19e-07   1.19e-05            9.54e-05
``float16``  9.77e-04   9.77e-02            7.81e-01
==========  =========  ==================  ==================
"""

import numpy as np

DEFAULT_SAFETY = 100.0
"""Unconditional minimum multiple of ``eps``, used when the conditioning is unknown."""

DEFAULT_CONDITIONING_SAFETY = 4.0
"""Multiplier applied to a supplied conditioning estimate."""


def working_dtype(solve_precision):
    """
    Normalise a precision token to a :class:`numpy.dtype`.

    Parameters
    ----------
    solve_precision : dtype-like or None
        Requested working precision. ``None`` means backend-native, i.e. ``float64``.

    Returns
    -------
    numpy.dtype
        The working precision.
    """
    return np.dtype('float64') if solve_precision is None else np.dtype(solve_precision)


def working_eps(solve_precision):
    """
    Machine epsilon of the working precision.

    Parameters
    ----------
    solve_precision : dtype-like or None
        Requested working precision.

    Returns
    -------
    float
        Machine epsilon.
    """
    return float(np.finfo(working_dtype(solve_precision)).eps)


def tol_floor(
    solve_precision, safety=DEFAULT_SAFETY, conditioning=1.0, conditioning_safety=DEFAULT_CONDITIONING_SAFETY
):
    r"""
    Smallest relative tolerance worth requesting at this working precision.

    Parameters
    ----------
    solve_precision : dtype-like or None
        Requested working precision.
    safety : float, optional
        Unconditional minimum multiple of the machine epsilon.
    conditioning : float, optional
        Estimate of :math:`1 + \alpha\|J\|` for the node-local system. Leave at 1 if unknown.
    conditioning_safety : float, optional
        Multiplier applied to ``conditioning``.

    Returns
    -------
    float
        The tolerance floor.
    """
    return working_eps(solve_precision) * max(safety, conditioning_safety * float(conditioning))


def clamp_tolerance(requested, solve_precision, safety=DEFAULT_SAFETY, conditioning=1.0):
    r"""
    Clamp a requested relative tolerance to the working-precision floor.

    Parameters
    ----------
    requested : float or None
        Requested relative tolerance. ``None`` selects the floor.
    solve_precision : dtype-like or None
        Requested working precision.
    safety : float, optional
        Unconditional minimum multiple of the machine epsilon.
    conditioning : float, optional
        Estimate of :math:`1 + \alpha\|J\|` for the node-local system.

    Returns
    -------
    tuple of (float, bool)
        The effective tolerance and whether it was clamped.
    """
    floor = tol_floor(solve_precision, safety, conditioning)
    if requested is None:
        return floor, False
    return (floor, True) if float(requested) < floor else (float(requested), False)


class PrecisionAwareTolerances:
    """
    Mixin supplying precision-aware Krylov and Newton tolerances.

    Call :meth:`setup_precision_tolerances` once during ``__init__``. Afterwards ``krylov_tol`` and
    ``newton_rtol`` hold tolerances the working precision can actually deliver, and
    ``tolerance_report`` documents what was requested and what was used.
    """

    def setup_precision_tolerances(self, solve_precision, krylov_tol, newton_rtol, safety=DEFAULT_SAFETY):
        """
        Derive effective tolerances from the working precision.

        Parameters
        ----------
        solve_precision : dtype-like or None
            Working precision of the node-local solve.
        krylov_tol : float or None
            Requested relative tolerance for the linear solver.
        newton_rtol : float or None
            Requested relative tolerance for the nonlinear solver.
        safety : float, optional
            Multiple of the machine epsilon used as the floor.

        Returns
        -------
        dict
            The tolerance report, also stored as ``self.tolerance_report``.
        """
        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        self.tolerance_safety = safety

        krylov_effective, krylov_clamped = clamp_tolerance(krylov_tol, self.solve_precision, safety)
        newton_effective, newton_clamped = clamp_tolerance(newton_rtol, self.solve_precision, safety)

        self.krylov_tol = krylov_effective
        self.newton_rtol = newton_effective
        self.tolerance_report = {
            'working_precision': str(working_dtype(self.solve_precision)),
            'eps': working_eps(self.solve_precision),
            'floor': tol_floor(self.solve_precision, safety),
            'krylov_requested': krylov_tol,
            'krylov_effective': krylov_effective,
            'krylov_clamped': krylov_clamped,
            'newton_requested': newton_rtol,
            'newton_effective': newton_effective,
            'newton_clamped': newton_clamped,
        }
        return self.tolerance_report

    def effective_tolerances(self, conditioning=1.0):
        r"""
        Re-clamp the requested tolerances for a known conditioning.

        The floor set in :meth:`setup_precision_tolerances` assumes nothing about the node-local
        system. Callers that know :math:`\alpha\|J\|` should use this instead, because the
        reachable relative tolerance scales with it.

        Parameters
        ----------
        conditioning : float, optional
            Estimate of :math:`1 + \alpha\|J\|` for the node-local system.

        Returns
        -------
        tuple of (float, float)
            Effective Krylov and Newton tolerances.
        """
        report = self.tolerance_report
        krylov, _ = clamp_tolerance(
            report['krylov_requested'], self.solve_precision, self.tolerance_safety, conditioning
        )
        newton, _ = clamp_tolerance(
            report['newton_requested'], self.solve_precision, self.tolerance_safety, conditioning
        )
        return krylov, newton

    def describe_tolerances(self):
        """
        Return a one-line human-readable summary of the tolerance policy.

        Returns
        -------
        str
            Summary string.
        """
        report = self.tolerance_report
        krylov_flag = ' (clamped)' if report['krylov_clamped'] else ''
        newton_flag = ' (clamped)' if report['newton_clamped'] else ''
        return (
            f"{report['working_precision']}: eps={report['eps']:.2e} floor={report['floor']:.2e} | "
            f"krylov {report['krylov_requested']} -> {report['krylov_effective']:.2e}{krylov_flag} | "
            f"newton {report['newton_requested']} -> {report['newton_effective']:.2e}{newton_flag}"
        )
