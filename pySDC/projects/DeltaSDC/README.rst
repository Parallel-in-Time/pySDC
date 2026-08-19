Delta-form SDC for mixed precision
==================================

This project implements SDC in *deferred-correction form* — the form in which the sweep solves for
a correction rather than for the solution — so that the expensive node-local spatial solve can run
entirely at reduced precision without capping the attainable accuracy.

Motivation
----------

Reducing the precision of the node-local solve is only safe if the quantity the solver returns is
**small**. A solver working at precision :math:`\varepsilon` delivers a relative accuracy
:math:`\varepsilon`; if it returns the state :math:`u = \mathcal{O}(1)`, that is an *absolute*
error of :math:`\varepsilon |u|` which the outer SDC iteration cannot remove, and the iteration
stalls at the reduced precision. If it returns a correction :math:`\delta`, the same relative
accuracy is an absolute error :math:`\varepsilon |\delta|`, and :math:`|\delta| \to 0` as the
sweeps converge, so full backend accuracy is retained.

The delta form makes that correction explicit.

Method
------

A standard SDC sweep

.. math::
    u^{k+1}_m = u_0 + \tau_m + \Delta t (Q f^k)_m
                + \Delta t \sum_j Q^\Delta_{mj}\,(f^{k+1}_j - f^k_j)

is algebraically identical to, writing :math:`\delta_m = u^{k+1}_m - u^k_m` and
:math:`\varepsilon_m = u_0 + \tau_m + \Delta t (Q f^k)_m - u^k_m` for the collocation residual,

.. math::
    \delta_m = \varepsilon_m + \Delta t \sum_j Q^\Delta_{mj}\,\Delta f_j,
    \qquad \Delta f_j = f(u^k_j + \delta_j) - f(u^k_j),
    \qquad u^{k+1}_m = u^k_m + \delta_m.

Every sweep is therefore already iterative refinement: a high-precision residual, a correction
solve, and a high-precision update. **No Jacobian appears**, so an IMEX splitting survives
unchanged, and nothing outside the sweeper needs to change — no core modifications, no controller
modifications, and multi-level and parallel-in-time runs are unaffected.

Contents
--------

===========================  ====================================================================
``sweepers.py``              ``delta_implicit`` and ``delta_imex_1st_order``
``precision.py``             precision-aware tolerance policy
``problems.py``              ``allencahn_delta``, an example nonlinear correction solve
``run_demo.py``              runnable demonstration
===========================  ====================================================================

Usage
-----

``delta_implicit`` is a drop-in replacement for ``generic_implicit`` and reproduces it exactly:

.. code-block:: python

    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    description['sweeper_class'] = delta_implicit

To get the precision benefit, the node-local solve has to be told to work on the correction. Which
route applies depends only on the implicit operator:

**Linear or affine implicit operator** (including IMEX with implicit diffusion) — nothing to
implement. Set ``linear_implicit=True`` in ``sweeper_params``. Since
:math:`f(w+\delta) - f(w) = A\delta`, the stock ``solve_system`` already solves the correction
equation once the affine part :math:`\alpha f(0,t)` has been removed, which the sweeper does.

**Nonlinear implicit operator** — the problem class must provide

.. code-block:: python

    def solve_system_delta(self, r, factor, base, f_base, t):
        """Solve  d - factor * [f(base + d) - f(base)] = r  and return d."""

``f_base`` is passed in because the sweeper already holds it, so honouring the contract costs no
extra right-hand side evaluation. Two properties are load-bearing:

* the solver must work on, and return, the **correction**;
* the increment :math:`f(w+\delta) - f(w)` must be formed **without cancellation**. Evaluating it
  as a difference of two :math:`\mathcal{O}(|f|)` quantities reinstates an absolute error of order
  :math:`\varepsilon |f|` and destroys the benefit. Expand it analytically instead, so that every
  term carries an explicit factor :math:`\delta`. See ``allencahn_delta`` for an example.

If neither route applies the sweeper falls back to the substitution :math:`y = u^k_m + \delta_m`.
That is always correct and identical to ``generic_implicit``, but the solver sees an
:math:`\mathcal{O}(1)` unknown, so there is no precision benefit.

Tolerances
----------

Asking a reduced-precision solver for a tolerance it cannot reach does not fail loudly — the solver
runs to ``maxiter`` against an impossible bar, turning reduced precision into a large
pessimisation. ``precision.py`` derives the floor from the working precision, clamps requested
tolerances to it, and records the clamping in ``tolerance_report`` so it is never silent.

Tolerances themselves stay ordinary problem parameters, set in the frontend exactly as elsewhere
in pySDC: ``lin_tol`` is the relative Krylov tolerance and ``newton_tol`` the absolute bar on the
correction residual, both keeping the meaning they have in the parent problem class. The only
thing this module adds is the **floor** below which a requested tolerance is unreachable, which is
**conditioning-aware**::

    tol_floor(dtype, conditioning) = eps(dtype) * max(safety, conditioning_safety * conditioning)

Forming the correction residual involves :math:`\alpha J \delta`, so its rounding error is of order
:math:`\varepsilon\,\alpha\|J\|\,|\delta|`: the smallest reachable relative tolerance scales with
:math:`\alpha\|J\|`, which grows with the step size and the spatial resolution. A fixed multiple of
``eps`` is too tight for stiff or well-resolved problems. This cannot live in the frontend:
:math:`\alpha = \Delta t Q^\Delta_{mm}` is only known inside the node-local solve and differs per
node and per level, so ``allencahn_delta`` raises the inherited tolerances to the floor per solve.
``safety`` is an unconditional minimum for callers that do not know :math:`\alpha\|J\|`.

A correction solve that exits on ``newton_maxiter`` rather than on its tolerance logs a warning,
since that is the signature of a tolerance the working precision cannot deliver.

Inexactness
-----------

The node-local solve is a preconditioner for the SDC iteration, not part of its fixed point, so it
need not be solved to full accuracy. The two knobs behave very differently, and the difference
matters:

* ``lin_tol`` is **relative**, and loosening it is close to free. Solving the linear system to a
  single digit still converges in the same number of sweeps to the same error, and costs about a
  third less inner work than solving it tightly. This is genuine inexact SDC — loosen it
  aggressively.
* ``newton_tol`` is **absolute**, and it caps the accuracy of the whole SDC iteration: the
  attainable error tracks it directly, so the residual can never fall much below it however many
  sweeps are run. Keep it well below the target accuracy.

The same distinction explains why reduced precision does not cap accuracy while an absolute
tolerance does. The precision floor is *relative* (``floor * |r|``), so it shrinks with the
correction; a fixed ``newton_tol`` does not.

A consequence worth knowing: while ``newton_tol`` sits above the precision floor, the effective bar
is ``newton_tol`` for every precision, so reduced- and full-precision runs do *identical* work and
return the same answer. Reduced precision only starts to differ once the requested tolerance drops
below what it can deliver.

Reduced-precision storage
-------------------------

``correction_precision`` additionally stores :math:`\varepsilon`, :math:`\delta` and
:math:`\Delta f` in a reduced-precision datatype built from the problem's own ``init`` tuple. This
is a secondary effect: the nodal state ``u`` and the right-hand sides ``f`` must stay in backend
precision, since the residual cancels :math:`\mathcal{O}(1)` quantities against each other.

Running
-------

.. code-block:: bash

    python pySDC/projects/DeltaSDC/run_demo.py
    pytest pySDC/projects/DeltaSDC/tests
