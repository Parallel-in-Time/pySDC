r"""
Precision-aware tolerance policy.

A solver working at precision :math:`\varepsilon` cannot drive a relative residual much below
:math:`\mathcal{O}(\kappa\varepsilon)`. Asking it to do so does not fail loudly: the solver simply
runs to ``maxiter`` against a bar it can never meet, which turns reduced precision into a large
pessimisation instead of a saving.

This module derives the tolerance floor from the working precision instead of inheriting a
precision-agnostic constant::

    tol_floor(dtype) = safety * eps(dtype)

with ``safety`` standing in for the condition number. Requested tolerances tighter than the floor
are clamped, and the clamping is recorded so it is never silent.

============  ===========  ===========
dtype         eps          floor (x100)
============  ===========  ===========
``float64``   2.22e-16     2.22e-14
``float32``   1.19e-07     1.19e-05
``float16``   9.77e-04     9.77e-02
============  ===========  ===========
"""

import numpy as np

DEFAULT_SAFETY = 100.0


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


def tol_floor(solve_precision, safety=DEFAULT_SAFETY):
    """
    Smallest relative tolerance worth requesting at this working precision.

    Parameters
    ----------
    solve_precision : dtype-like or None
        Requested working precision.
    safety : float, optional
        Multiple of the machine epsilon used as the floor.

    Returns
    -------
    float
        The tolerance floor.
    """
    return safety * working_eps(solve_precision)


def clamp_tolerance(requested, solve_precision, safety=DEFAULT_SAFETY):
    """
    Clamp a requested relative tolerance to the working-precision floor.

    Parameters
    ----------
    requested : float or None
        Requested relative tolerance. ``None`` selects the floor.
    solve_precision : dtype-like or None
        Requested working precision.
    safety : float, optional
        Multiple of the machine epsilon used as the floor.

    Returns
    -------
    tuple of (float, bool)
        The effective tolerance and whether it was clamped.
    """
    floor = tol_floor(solve_precision, safety)
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
