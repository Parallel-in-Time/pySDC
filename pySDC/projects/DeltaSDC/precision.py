r"""
Working-precision tolerance floor.

Tolerances themselves are ordinary problem parameters and are set in the frontend, exactly as
everywhere else in pySDC — ``lin_tol`` for the Krylov solver, ``newton_tol`` for the nonlinear one.
This module adds only the one thing the frontend cannot express: the *floor* below which a
requested tolerance is unreachable at a given working precision.

Asking a solver for a tolerance it cannot reach does not fail loudly. It runs to ``maxiter``
against an impossible bar, which turns reduced precision into a large pessimisation instead of a
saving.

The floor depends on the conditioning of the node-local system, not only on the machine epsilon.
Forming the correction residual :math:`g = \delta - \alpha[f(w+\delta) - f(w)] - r` involves the
term :math:`\alpha J \delta`, so its rounding error is of order
:math:`\varepsilon\,\alpha\|J\|\,|\delta|`. The smallest reachable *relative* tolerance therefore
scales with :math:`\alpha\|J\|`, which grows with the step size and with the spatial resolution::

    tol_floor(dtype, conditioning) = eps(dtype) * max(safety, conditioning_safety * conditioning)

This is why the floor cannot live in the frontend: :math:`\alpha = \Delta t Q^\Delta_{mm}` is only
known inside the node-local solve, and differs per node and per level. Callers that know
:math:`\alpha\|J\|` pass ``conditioning=1 + alpha * norm``; ``safety`` is an unconditional minimum
for callers that do not.

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
    Raise a requested relative tolerance to the working-precision floor if it is below it.

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
