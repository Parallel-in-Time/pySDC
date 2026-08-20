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
``sweepers_MPI.py``          ``delta_implicit_MPI``, one collocation node per rank
``problems.py``              ``allencahn_delta``, an example nonlinear correction solve
``problems_petsc.py``        ``petsc_fisher_delta`` (reduced precision emulated)
``problems_fenics.py``       ``fenics_grayscott_delta`` (reduced precision emulated)
``run_demo.py``              runnable demonstration
``run_petsc.py``             PETSc entry point, called by the ``petsc``-marked test
``run_fenics.py``            FEniCS entry point, called by the ``fenics``-marked test
``run_mpi.py``               node-parallel driver, spawned by the ``mpi4py``-marked test
===========================  ====================================================================

The optional-backend modules are imported separately so ``sweepers`` and ``problems`` stay usable
without ``mpi4py``, ``petsc4py`` or ``dolfin``.

Coverage
--------

===========================  ====================================================================
SDC, MLSDC (2 and 3 levels)  covered
PFASST (multi-level, multi-step)  covered
MSSDC (single level, multi-step)  covered
Node-parallel sweeper        covered, spawns ``mpirun`` with one rank per node
IMEX                         covered
PETSc                        covered, reduced precision **emulated**
FEniCS                       covered, linear via ``fenics_heat``, nonlinear via Gray-Scott
===========================  ====================================================================

PETSc and DOLFIN fix their scalar type at build time, so genuine single precision needs a
``--with-precision=single`` build. Both backends therefore *emulate* it: values are rounded through
the requested precision while the arithmetic stays at the backend type. That caps the information
but is optimistic about iteration counts and attainable accuracy, and results obtained this way
should be labelled as emulated.

Backend tests live under ``pySDC/tests/test_projects/test_DeltaSDC`` rather than in this project's
own ``tests`` folder, because the CI job that installs FEniCS, PETSc and mpi4py selects tests by
marker while the project job installs only this project's environment. Same split as
``pySDC/tests/test_tutorials/test_step_7.py``: the logic lives in the ``run_*.py`` scripts here and
the tests just call them.

Two limitations the FEniCS backend exposed:

* ``linear_implicit=True`` additionally requires the implicit solve to be **homogeneous in its
  boundary conditions**. It reuses the stock ``solve_system`` to solve the correction equation, and
  ``fenics_heat.solve_system`` applies inhomogeneous Dirichlet data to whatever right-hand side it
  is handed, which imposes the wrong boundary values on a correction. The substitution fallback is
  exact there and is what the FEniCS entry point uses.
* ``correction_precision`` needs a datatype that can be built at another precision. A
  ``fenics_mesh`` is backed by a DOLFIN function and cannot be, so requesting it raises a clear
  ``NotImplementedError`` rather than failing obscurely.

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

Tolerances are ordinary problem parameters, set in the frontend exactly as elsewhere in pySDC:
``lin_tol`` is the relative Krylov tolerance and ``newton_tol`` the absolute bar on the correction
residual, both keeping the meaning they have in the parent problem class.

A problem implementing ``solve_system_delta`` must additionally raise them to what its working
precision can deliver. Asking a reduced-precision solver for an unreachable tolerance does not fail
loudly — it runs to ``newton_maxiter`` against an impossible bar. Measured on the demo
configuration, removing this floor costs **20x the linear work in float32** for an identical
answer, so it is not optional::

    floor = eps(dtype) * max(100, 4 * (1 + alpha * ||J||))
    krylov_tol = max(lin_tol, floor)
    newton_bar = max(newton_tol, floor * |r|)

The floor scales with :math:`\alpha\|J\|` because forming the correction residual involves
:math:`\alpha J \delta`, so its rounding error is of order
:math:`\varepsilon\,\alpha\|J\|\,|\delta|`. That is also why it cannot live in the frontend:
:math:`\alpha = \Delta t Q^\Delta_{mm}` is only known inside the node-local solve and differs per
node and per level. ``allencahn_delta`` shows the four lines this takes; a bound on
:math:`\|J\|_\infty` computed once at construction is enough.

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
