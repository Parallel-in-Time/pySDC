Delta-form SDC for mixed precision
==================================

SDC in *deferred-correction form* -- the sweep solves for a correction rather than for the solution
-- so that the expensive parts of the iteration can run at reduced precision without capping the
attainable accuracy.

Motivation
----------

Reducing the precision of a node-local solve is only safe if the quantity the solver returns is
**small**. A solver working at precision :math:`\varepsilon` delivers a relative accuracy
:math:`\varepsilon`; if it returns the state :math:`u = \mathcal{O}(1)`, that is an *absolute* error
:math:`\varepsilon|u|` which the outer iteration cannot remove, and the iteration stalls at the
reduced precision. If it returns a correction :math:`\delta`, the same relative accuracy is an
absolute error :math:`\varepsilon|\delta|`, and :math:`|\delta| \to 0` as the sweeps converge, so
full backend accuracy is retained.

The delta form makes that correction explicit, and the same argument then applies one level up: a
coarse level is a preconditioner whose returned correction also tends to zero.

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
unchanged, and nothing outside the sweeper changes -- no core and no controller modifications, and
multi-level and parallel-in-time runs are unaffected.

Usage
-----

``delta_implicit`` is a drop-in replacement for ``generic_implicit`` and reproduces it exactly:

.. code-block:: python

    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    description['sweeper_class'] = delta_implicit

To get the precision benefit, the node-local solve has to be told to work on the correction. Which
route applies depends only on the implicit operator:

**Linear or affine implicit operator** (including IMEX with implicit diffusion) -- nothing to
implement. Set ``linear_implicit=True`` in ``sweeper_params``. Since
:math:`f(w+\delta) - f(w) = A\delta`, the stock ``solve_system`` already solves the correction
equation once the affine part :math:`\alpha f(0,t)` has been removed, which the sweeper does.

**Nonlinear implicit operator** -- the problem class provides

.. code-block:: python

    def solve_system_delta(self, r, factor, base, f_base, t):
        """Solve  d - factor * [f(base + d) - f(base)] = r  and return d."""

``f_base`` is passed in because the sweeper already holds it. Two properties are load-bearing: the
solver must work on and return the **correction**, and the increment :math:`f(w+\delta) - f(w)` must
be formed **without cancellation** -- expanded analytically, so every term carries an explicit factor
:math:`\delta`, rather than as a difference of two :math:`\mathcal{O}(|f|)` quantities. See
``allencahn_delta``.

If neither route applies the sweeper falls back to the substitution :math:`y = u^k_m + \delta_m`.
That is always correct and identical to ``generic_implicit``, but the solver sees an
:math:`\mathcal{O}(1)` unknown, so there is no precision benefit.

For a multi-level run the sweeper does not change -- swap the **transfer**:

.. code-block:: python

    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    description['base_transfer_class'] = delta_transfer

``BaseTransfer`` lets every level rebuild its own residual, which is what stock MLSDC does;
``delta_transfer`` hands a coarse level the restricted fine residual instead, and takes back the
corrections it accumulated. The sweeper does whichever it is given.

Contents
--------

The delta form and its hierarchy are ordinary pySDC, under ``pySDC/implementations``:

============================================  =======================================
``sweeper_classes/delta_form.py``             the sweepers, at any number of levels
``sweeper_classes/delta_form_MPI.py``         the node-parallel counterparts
``transfer_classes/BaseTransferDelta.py``     the hierarchy's transfer
``transfer_classes/BaseTransferDeltaMPI.py``  its node-parallel counterpart
============================================  =======================================

The two optional problem methods they dispatch on, ``eval_f_increment`` and ``solve_system_delta``,
are documented on :class:`pySDC.core.problem.Problem`. The hierarchy is an algebraic rewrite of
MLSDC, verified against :class:`BaseTransfer` for two, three and four levels and for PFASST, and
slightly cheaper, because the FAS :math:`\tau` is then never built.

What stays in this project is the mixed-precision study -- the emulation, the measurements and the
controls:

===========================  ====================================================================
``mlsdc.py``                 reduced precision on a level, emulated
``cascade.py``               storage precision raised as the iteration converges
``sweepers_MPI.py``          the same, for the node-parallel hierarchy
``problems.py``              ``allencahn_delta`` (nonlinear) and ``heat_delta`` (linear)
``problems_petsc.py``        ``petsc_fisher_delta`` (reduced precision emulated)
``problems_fenics.py``       ``fenics_grayscott_delta`` (emulated), plus the controls
``run_demo.py``              runnable demonstration, nonlinear and linear
``run_petsc.py``             PETSc entry point, called by the ``petsc``-marked test
``run_fenics.py``            FEniCS entry point, called by the ``fenics``-marked test
``run_mpi.py``               node-parallel driver, spawned by the ``mpi4py``-marked test
``paradiag.py``              ParaDiag at reduced precision -- no reformulation needed
===========================  ====================================================================

The names from ``pySDC/implementations`` are re-exported here with the emulation layered on, so
``mlsdc.delta_transfer`` is the ported transfer plus rounding. The optional-backend modules are
imported separately so ``mlsdc`` and ``problems`` stay usable without ``mpi4py``, ``petsc4py`` or
``dolfin``.

What can run at reduced precision
---------------------------------

Three independent places, with different requirements:

=========================  ===========================  ======================================
place                      how low                      why
=========================  ===========================  ======================================
node-local solve           see *delivered accuracy*     it returns a correction
whole coarse level         ``float16``                  it is a preconditioner one level up
fine level state, early    ``float16``, then climbing   early roundings are contracted away
fine level state, late     ``float64``, always          the residual cancels O(1) quantities
=========================  ===========================  ======================================

They compose: the whole stack lands within one iteration of the ``float64`` baseline.

The node-local solve
~~~~~~~~~~~~~~~~~~~~

The iteration sees exactly two things about a node-local solve: the **relative accuracy**
:math:`\eta` of the correction handed back, and whether the format it comes back in can represent
it. Not the algorithm, not the working precision, not the inner iteration count. So the useful
specification is :math:`\eta`, measured by solving exactly and then spoiling the answer by a random
relative error of that size (``tests/inexact_problem.py``):

==================================  ======  ======  ======  ======  ======  ======
solver delivers (relative)          1e-8    1e-6    1e-4    1e-3    1e-2    1e-1
==================================  ======  ======  ======  ======  ======  ======
SDC, fine solve                     14      14      15      16      18      39
MLSDC, fine solve                   7       8       9       11      16      stalls
MLSDC, coarse solve                 7       7       7       8       8       14
MLSDC, both solves                  7       8       9       11      16      stalls
==================================  ======  ======  ======  ======  ======  ======

Reading off the point where one iteration is lost:

* **SDC, fine solve: about four digits**, so ``float16`` at one extra iteration;
* **MLSDC, fine solve: about six digits**, so ``float32``, and that is the floor;
* **MLSDC, coarse solve: about two digits**, so ``float16`` for free.

Four orders of magnitude between the fine and the coarse solve *in the same run*. And MLSDC's fine
solve needs roughly a hundred times more accuracy than SDC's, because a method that contracts twice
as fast absorbs proportionally less solver error per sweep: MLSDC does not buy a sloppier fine
solve, it buys a coarse level to be sloppy on.

**How the solver reaches** :math:`\eta` **is its own business.** Half- or quarter-precision kernels
with iterative refinement on top, a low-precision factorisation used as a preconditioner, a loose
Krylov tolerance, a few multigrid cycles -- all invisible to the iteration. The format rows in the
next section are one way of reaching a delivered accuracy, not a constraint on the solver's
internals; the way the emulation reaches it, by rounding the operator and solving that perturbed
system exactly, is the **un-refined** path, which is *stricter* than the delivered accuracy alone
because the operator perturbation is amplified by the conditioning of the system.

Where refinement is worth paying for follows from the table. The coarse solve needs two digits,
which is a single low-precision solve with no inner refinement at all; refining it buys nothing,
because the outer iteration is itself iterative refinement and absorbs what is left in a coarse
correction on the next fine sweep. The fine solve under MLSDC needs six digits, and that is where
refinement from low-precision kernels earns its keep.

Measured on 1D heat, 127 → 63, :math:`\Delta t = 10^{-1}`, and asserted in
``tests/test_emulation.py::test_delivered_accuracy_requirement``. The ordering -- coarse four orders
looser than fine -- is the robust part; the exact thresholds are this configuration.

A whole coarse level
~~~~~~~~~~~~~~~~~~~~

Stock MLSDC cannot run a coarse level below backend precision, because it forms two differences of
:math:`\mathcal{O}(1)` quantities per V-cycle: the coarse residual, rebuilt from coarse data as
:math:`\varepsilon_G = u_G[0] + \tau + \Delta t (Q f_G) - u_G[m]`, and the coarse-grid correction,
recovered in ``BaseTransfer.prolong`` as :math:`u_G - u_G^{\mathrm{old}}`. Each carries an absolute
error :math:`\varepsilon|u|` that does not shrink as the iteration converges. Both have exact
algebraic replacements:

.. math::
    \varepsilon_G = R\,\varepsilon_F,
    \qquad
    u_G - u_G^{\mathrm{old}} = \sum_{\text{sweeps}} \delta_G .

The first is what the FAS :math:`\tau` is *for*: substituting
:math:`\tau = R(\Delta t\,Q_F f_F) - \Delta t\,Q_G f_G` into the coarse residual cancels the
:math:`\Delta t\,Q_G f_G` terms identically and leaves the restricted fine residual. The second is
already computed by the sweep. The delta hierarchy uses both, and both are tested against the
quantities they replace. The residual then travels with the level, advanced by a sweep as

.. math::
    \varepsilon \leftarrow \varepsilon - \delta + \Delta t\,(Q\,\Delta f),

and by a prolongation with :math:`\delta` the interpolated coarse correction. Both combine only
small quantities, which is what lets a middle level be swept again on the way up without rebuilding
an :math:`\mathcal{O}(1)` residual. :math:`\tau` is then never read at all, and neither is
:math:`u_G`, except as the base state a node-local increment is taken around.

A level below backend precision additionally needs **an analytic increment** and **a real correction
route**:

.. code-block:: python

    def eval_f_increment(self, base, delta, t):
        """f(base + delta) - f(base), with every term carrying a factor delta."""

Without it the sweeper forms :math:`\Delta f` by subtracting two stored right-hand sides, whose error
:math:`\varepsilon|f| = \varepsilon|Au|` carries the operator norm; that binds before the residual or
the solve does. Dropping it floors the heat run at 3.4e-06 with a ``float32`` coarse level, and on
the FEniCS heat equation, where :math:`|M^{-1}Ku| \approx 10^5`, the subtraction is already 7.7e-11
off *in double precision*.

And either ``solve_system_delta`` or ``linear_implicit=True`` is needed, because the substitution
fallback -- though exact -- reads the level's :math:`\mathcal{O}(1)` state: it adds :math:`u^k_m`,
subtracts :math:`\alpha f^k_m`, and recovers :math:`\delta` by subtracting :math:`u^k_m` again. At
backend precision that is merely no benefit; below it, it is a wrong answer -- 1.4e-05 on FEniCS heat
with a ``float32`` coarse level, 1.5e-04 on node-parallel heat.

Measured on 1D heat, 127 → 63, :math:`\Delta t = 10^{-1}`, ``LU``. Plain SDC needs **14** iterations
here and MLSDC **7**, so the coarse level is doing real work and losing it would show immediately:

====================================  ==========  ==============
configuration                         iterations  residual floor
====================================  ==========  ==============
everything ``float64``                7           8.0e-14
coarse *solve* ``float32``            7           9.1e-14
coarse *solve* ``float16``            7           7.4e-14
coarse *level* ``float32``            7           8.0e-14
coarse level and solve ``float16``    7           7.4e-14
+ coarse corrections ``float16``      10          9.3e-14
+ fine solve ``float32``              10          8.0e-14
CONTROL stock hierarchy, fp32 coarse  never       **2.7e-10**
CONTROL stock hierarchy, fp16 coarse  never       **2.3e-04**
CONTROL fine *level* ``float32``      never       **2.2e-05**
CONTROL fine solve fp16, unscaled     never       **5.7e-05**
====================================  ==========  ==============

The four controls are what make the rest of the table mean anything. The first two say the delta
*hierarchy* is what makes a reduced-precision coarse level safe, not the node-local solve; the third
says the fine level is where precision binds; the fourth is the scaling below.

The same holds for three and four levels, for PFASST, for the node-parallel sweeper, and on every
backend. ``run_demo.py``, ``run_petsc.py``, ``run_fenics.py`` and ``run_mpi.py`` each print their own
version of this matrix and assert it.

Corrections below ``float16`` need scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The smallest ``float16`` subnormal is 6e-8. The delta form exists to make the solver's argument
small, so the two are in direct tension: a correction of 1e-10 handed to a half-precision solver
rounds to **zero**. Scaling the right-hand side to :math:`\mathcal{O}(1)` before the solve and
scaling the result back -- exact for a linear solve, and free, since :math:`|r|` is already known --
removes it.

``correction_precision``, which stores :math:`\varepsilon`, :math:`\delta` and :math:`\Delta f` in a
reduced-precision datatype, gets the same treatment: the stored quantities are divided by the
residual's own magnitude before they are written and multiplied back on the way out. That is block
floating point, and what half-precision hardware does anyway. One divisor serves the whole sweep, so
ordinary arithmetic on the stored quantities stays correct. Measured on plain SDC, whose corrections
run all the way down to 1e-13:

=======================================  ==========  ==============
corrections on the fine level stored at  iterations  residual floor
=======================================  ==========  ==============
``float64``                              14          8.6e-14
``float32``                              14          7.6e-14
``float16``, scaled                      15          9.5e-14
``float16``, unscaled                    never       **5.9e-05**
=======================================  ==========  ==============

5.9e-05 is ``float16``'s smallest *normal*, 6.1e-05. Below it half precision has almost no mantissa
left -- 1.3e-2 relative at 1e-6, 1.9e-1 at 1e-7 -- so an unscaled correction turns to noise exactly
when it starts to matter.

On the *fine* solve, unscaled half precision floors the iteration at 5.7e-05; scaled, it converges at
three extra iterations. On a *coarse* solve the same underflow costs iterations only, 10 instead of
7, with the floor unchanged. Losing the coarse correction degrades MLSDC towards SDC; losing the
fine correction stalls it. That is the precise sense in which a coarse level tolerates less
precision than a fine one -- it fails gracefully, which the fine level does not.

The fine level's state, as the iteration converges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"The fine level's state must be ``float64``" is a statement about the **end** of the run. Early on
the residual is nowhere near :math:`\varepsilon|u|`, and the rounding a low format introduces is not
damage that persists -- it is a perturbed iterate, which the remaining sweeps contract away like any
other. An error injected at sweep :math:`k` is damped by :math:`\rho^{N-k}`.

So the state can walk up a ladder, ``float16 -> float32 -> float64``, and the delta form supplies its
own indicator for when to step. Rounding to a format perturbs the state by :math:`\varepsilon|u|`;
the sweep just applied a correction of size :math:`|\delta|`; if the perturbation is not comfortably
smaller, the update is lost in the rounding and the format is spent:

.. math::
    \text{use the lowest format with}\quad |\delta| > \texttt{safety}\cdot\varepsilon\,|u| .

Both norms are already in hand. No operator norm, no :math:`\Delta t`, nothing problem-specific and
nothing to calibrate. Measured on 1D heat, 127 → 63:

=============================  ==========  ===============  ==========================
configuration                  iterations  diff to float64  sweeps per format
=============================  ==========  ===============  ==========================
SDC, float64 throughout        14          --               64: 14
SDC, cascade, safety 10        14          2.9e-15          16: 3, 32: 5, 64: 6
SDC, cascade, safety 100       14          3.7e-16          16: 2, 32: 4, 64: 8
SDC, cascade, safety 1000      14          1.7e-16          32: 5, 64: 9
MLSDC, float64 throughout      7           --               64: 7
MLSDC, cascade, safety 100     8           1.0e-13          16: 1, 32: 2, 64: 5
MLSDC, cascade, safety 1000    7           2.7e-16          32: 2, 64: 5
=============================  ==========  ===============  ==========================

``safety = 100`` is the default and the one setting that stays within one iteration of both
baselines. SDC then runs six of its fourteen sweeps below backend precision for nothing; MLSDC runs
three of eight, which is the same story as everywhere else here -- it converges too fast to have
much early slack to sell.

Two rules the cascade obeys. It is walked **monotonically**, so a run pays at most
``len(ladder) - 1`` conversions however many iterations it takes; and it may only lower the precision
of quantities the iteration will later correct, which excludes ``u[0]`` -- see *Lessons learned*.

Every example carries a cascade row: Allen-Cahn matches its own baseline sweep for sweep to 1.9e-14,
PETSc Fisher to 9.7e-13, FEniCS Gray-Scott costs one sweep of fifteen, and the node-parallel driver
agrees to 7.0e-12 with the indicator reduced across the node communicator, since a rank that stepped
while its neighbours did not would leave the level at two precisions at once.

All three together
~~~~~~~~~~~~~~~~~~

One division has to be respected: the **cascade is for the finest level only**, the one level whose
requirement is absolute and so the one with somewhere to climb to. A coarse level is a
preconditioner, its requirement is relative, and it belongs at a fixed low precision. With that
respected, on 1D heat 127 → 63:

=================================================  ==========  ========
configuration                                      iterations  diff
=================================================  ==========  ========
SDC, float64 throughout                            14          --
SDC, fine solve float32                            14          4.1e-16
SDC, fine solve float32 + cascade                  14          9.7e-16
SDC, fine solve float16 + cascade                  15          9.4e-14
MLSDC, float64 throughout                          7           --
MLSDC, coarse level and solve float16              7           5.6e-15
MLSDC, + fine solve float32                        7           5.8e-15
MLSDC, + fine-state cascade (all three)            8           1.0e-13
MLSDC, all three but fine solve float16            **10**      1.0e-13
=================================================  ==========  ========

The whole stack is one iteration off the baseline, and the fixed-precision parts of it are free. The
last row is where it stops, and it stops exactly where the delivered-accuracy specification says it
should: MLSDC's fine solve needs about six digits and ``float16`` carries three.

Reduced precision that is not emulated
--------------------------------------

The tables above are measured with values rounded through a format while the arithmetic and the
containers stay at the backend type. That bounds the information honestly but saves no memory and
cannot be timed, so the finite-difference problems take a ``dtype``:

.. code-block:: python

    problem_params['dtype'] = ['float64', 'float16']   # per level, as usual

The level's arrays then genuinely are ``float16``. The operators follow at
``promote_types(dtype, float32)``, because SciPy holds no half-precision sparse matrix and hardware
that stores half precision computes in single anyway -- so ``float16`` means genuinely
half-precision *storage* with single-precision arithmetic, which is the arrangement it has on real
hardware too. ``mesh_to_mesh`` carries its transfer operators at the precision of what they produce,
for the same reason.

Same configuration as above, 1D heat 127 → 63, measured on NumPy 2:

=========================================  ==========  =========  ===========  =======
coarse level                               iterations  residual   u+f bytes    diff
=========================================  ==========  =========  ===========  =======
``float64``                                7           2.8e-12    12160        --
genuine ``float32``                        7           2.8e-12    10144        3.5e-16
genuine ``float16``                        11          9.6e-12    9136         1.7e-13
emulated, ``level_precision`` only         7           2.8e-12    12160        0.0
emulated, level *and* correction           10          1.3e-12    12160        1.2e-13
=========================================  ==========  =========  ===========  =======

**Single precision on a coarse level is free, and now actually smaller.** That claim survives contact
with a real dtype unchanged. **Half precision is not free: it costs four iterations, or three under
NumPy 1's promotion rule.** The emulation only predicts the cost once ``correction_precision`` is set
alongside ``level_precision``, because a level that genuinely stores at ``float16`` also holds its
corrections there. So the rule for reading the emulated tables: they are trustworthy to within one
iteration where both storage knobs are set together, and optimistic where only ``level_precision``
is.

The memory win is bounded by the fine level, which has to stay ``float64``: 17 % of the stored state
at ``float32``, 25 % at ``float16``, for a two-level hierarchy where the fine level is twice the size
of the coarse one. Timing it still needs hardware that carries the format.

``level_precision`` stays for the backends that cannot take a ``dtype``: PETSc and DOLFIN fix their
scalar type at build time, so there is nothing to pass them. It is a measurement instrument for
those, not a mechanism.

ParaDiag
--------

ParaDiag needs none of the above. It is already in the delta form, because the ParaDiag iteration
*is* iterative refinement over the whole block:

.. math::
    r^k = b - \mathcal{C}u^k, \qquad
    \delta^k = \mathcal{C}_\alpha^{-1} r^k, \qquad
    u^{k+1} = u^k + \delta^k.

The residual is formed in working precision, handed to the diagonalised alpha-circulant
preconditioner, and the result added back -- so the quantity the node-local solver receives and
returns is already an increment whose magnitude falls with the iteration. ``paradiag.py`` therefore
contains no reformulation, only two precision knobs and the tests that say what they cost. Measured
on 8 steps of 1D heat, 3 nodes, :math:`\alpha = 10^{-4}`:

==================================  ==============  ===========
configuration                       residual floor  iterations
==================================  ==============  ===========
everything ``complex128``           6.3e-16         4
node-local solve ``complex64``      6.1e-16         5
weighted FFT ``complex64``          6.5e-16         5
whole preconditioner ``complex64``  8.9e-16         5
*solution storage* ``complex64``    **9.6e-08**     30 (stalls)
==================================  ==============  ===========

Everything inside the preconditioner is free; the last row is the control, and the only place
precision binds. The cost appears as rate, not accuracy: one extra iteration, and only at small
:math:`\alpha`, which is the signature of a perturbed preconditioner.

The FFT row is worth a note. The weighted transform is deliberately ill-conditioned --
``get_J_inv_matrix`` weights entry :math:`l` by :math:`\alpha^{l/L}`, so the inverse amplifies by up
to :math:`1/\alpha`, and the expected floor is :math:`\varepsilon/\alpha`, about 1e-3 here. It does
not happen, because the amplification acts on the increment, which is itself shrinking. The
conditioning of the transform limits how small :math:`\alpha` may be for a given *state* precision;
it does not limit the precision of the transform.

Two practical notes. ParaDiag diagonalises in time, so its working type is complex and reduced
precision means ``complex64``, not ``float32``. And unlike the PETSc and FEniCS routes, nothing here
is emulated: SciPy's sparse solver and NumPy's matmul both carry ``complex64`` through.

Tolerances and inexactness
--------------------------

Tolerances are ordinary problem parameters, set in the frontend exactly as elsewhere in pySDC:
``lin_tol`` is the relative Krylov tolerance and ``newton_tol`` the absolute bar on the correction
residual. The two behave very differently, and the difference matters:

* ``lin_tol`` is **relative**, and loosening it is close to free. Solving the linear system to a
  single digit still converges in the same number of sweeps to the same error, and costs about a
  third less inner work. This is genuine inexact SDC -- loosen it aggressively.
* ``newton_tol`` is **absolute**, and it caps the accuracy of the whole SDC iteration: the attainable
  error tracks it directly. Keep it well below the target accuracy.

The same distinction explains why reduced precision does not cap accuracy while an absolute tolerance
does. The precision floor is *relative*, so it shrinks with the correction; a fixed ``newton_tol``
does not. A consequence worth knowing: while ``newton_tol`` sits above the precision floor, the
effective bar is ``newton_tol`` for every precision, so reduced- and full-precision runs do
*identical* work and return the same answer.

A problem implementing ``solve_system_delta`` must raise its tolerances to what its working precision
can deliver. Asking a reduced-precision solver for an unreachable tolerance does not fail loudly --
it runs to ``newton_maxiter`` against an impossible bar, which costs **20x the linear work in
float32** for an identical answer::

    floor = eps(dtype) * max(100, 4 * (1 + alpha * ||J||))
    krylov_tol = max(lin_tol, floor)
    newton_bar = max(newton_tol, floor * |r|)

The floor scales with :math:`\alpha\|J\|` because forming the correction residual involves
:math:`\alpha J \delta`. That is also why it cannot live in the frontend:
:math:`\alpha = \Delta t Q^\Delta_{mm}` is only known inside the node-local solve and differs per
node and per level. ``allencahn_delta`` shows the four lines this takes. A correction solve that
exits on ``newton_maxiter`` rather than on its tolerance logs a warning.

Coverage
--------

================================  ===============================================================
SDC                               covered
MLSDC (2, 3 and 4 levels)         covered, in delta form of the hierarchy
PFASST (multi-level, multi-step)  covered
MSSDC (single level, multi-step)  covered
Node-parallel sweeper             covered, single- and multi-level, one rank per node
IMEX                              covered, single- and multi-level
ParaDiag                          covered, reduced precision **genuine**, not emulated
PETSc                             covered, reduced precision **emulated**
FEniCS                            covered, linear via ``fenics_heat``, nonlinear via Gray-Scott
================================  ===============================================================

PETSc and DOLFIN fix their scalar type at build time, so genuine single precision needs a
``--with-precision=single`` build. Both backends therefore emulate it, which caps the information but
is optimistic about iteration counts; results obtained this way should be labelled as emulated.

Backend tests live under ``pySDC/tests/test_projects/test_DeltaSDC`` rather than in this project's own
``tests`` folder, because the CI job that installs FEniCS, PETSc and mpi4py selects tests by marker
while the project job installs only this project's environment. Same split as
``pySDC/tests/test_tutorials/test_step_7.py``: the logic lives in the ``run_*.py`` scripts here and
the tests just call them.

Running
-------

.. code-block:: bash

    python pySDC/projects/DeltaSDC/run_demo.py
    pytest pySDC/projects/DeltaSDC/tests

Lessons learned and dead ends
-----------------------------

Traps, near-misses and roads not taken. None of this is needed to use the project.

**A small-step run will tell you the reformulation is unnecessary.** Stock MLSDC survives a
reduced-precision coarse level in some configurations, because the prolonged rounding noise is
grid-scale and an ``LU`` sweep on a stiff operator largely annihilates it. It degrades exactly where
the coarse level starts to matter, and it depends on :math:`\Delta t` -- same code, same problem:

=========================  ==================  ==================
config                     stock, fp32 coarse  delta, fp32 coarse
=========================  ==================  ==================
63 → 31, :math:`dt=0.01`   3.8e-15             3.8e-15
63 → 31, :math:`dt=0.1`    4.2e-08             1.8e-14
127 → 63, :math:`dt=0.01`  5.2e-14             1.8e-14
127 → 63, :math:`dt=0.1`   5.0e-09             9.1e-14
127 → 63, :math:`dt=1.0`   2.7e-09             1.4e-13
=========================  ==================  ==================

**Widen first, scale second.** Multiplying a stored value by its scale *before* widening it does the
multiply at the reduced precision, and a scale of 1e-8 then puts the result below ``float16``'s
smallest subnormal on the way out. That mistake and a missing scale look identical from outside --
both stall the iteration around 1e-4 -- which is why the round trip has a test of its own that walks
the magnitude from 1e0 down to 1e-16.

**A missing** ``float()`` **silently restores float64.** A coefficient taken out of a numpy array is
an ``np.float64``, and under NEP 50 -- NumPy 2's rule -- multiplying a reduced-precision array by one
produces ``float64``. Nothing fails and no answer changes; the level is just no longer reduced. The
sweepers and the transfer coerce every collocation coefficient, and a test asserts that every array a
reduced-precision level holds is still at that precision when the run ends. ``NPY_PROMOTION_STATE=weak``
reproduces the NumPy 2 rule on NumPy 1.26, which is how the coercions are checked to be load-bearing.

**The cascade must not round** ``u[0]``. It and the right-hand side at it are input data, not
iterates: no sweep corrects them, so degrading them perturbs the problem rather than the
approximation to it, and raising the precision afterwards does not undo it. Rounding them converges
just as nicely to an answer 4.0e-06 out with a residual of 2.7e-12 -- a failure a residual-based
check does not see.

**The cascade needs a run that converges.** It climbs as the corrections shrink, so a run stopped at
a fixed, short iteration count ends *while still in a low format* and returns that answer. The FEniCS
heat table runs six iterations and compares bit-for-bit; a cascade row there came out 4.0e-07 wrong
and is deliberately absent.

**Stagnation detection is the better indicator and does not fit.** Switching when a sweep fails to
contract needs no estimate at all and gives SDC seven low-precision sweeps of fourteen instead of
six. But it needs iterations to react -- two to establish a healthy rate, one more to see it break --
which is affordable in fourteen sweeps and not in seven, so it costs MLSDC two iterations. The
:math:`|\delta|` indicator fires immediately at the price of being conservative.

**PFASST still differences one pair of** :math:`\mathcal{O}(1)` **values.** The initial value arrives
from the predecessor *after* the restriction, straight into ``u[0]``, and the inherited residual has
to follow it: :math:`\varepsilon_m \leftarrow \varepsilon_m + (u_0 - u_0^{\mathrm{ref}})`. The
difference is small and converges to zero, but it is *formed* by cancelling two values of size
:math:`|u|`, so a reduced-precision coarse level buys less under PFASST than under MLSDC. Putting the
step-to-step exchange itself in delta form would remove it; that is not done here.

**FEniCS needs its own correction solve.** ``linear_implicit=True`` additionally requires the implicit
solve to be homogeneous in its boundary conditions, and ``fenics_heat.solve_system`` applies
inhomogeneous Dirichlet data to whatever right-hand side it is handed, which imposes the wrong
boundary values on a correction. ``fenics_heat`` therefore carries ``solve_system_delta`` itself --
the same system with ``bc_hom`` applied. Separately, ``correction_precision`` needs a datatype that
can be built at another precision, and a ``fenics_mesh`` cannot be, so requesting it raises
``NotImplementedError``.

**A genuine cascade is not built.** The cascade rounds in place, which is right for an emulation and
wrong for a real format: raising the precision of a genuinely ``float16`` level means *reallocating*
its arrays at the new dtype. ``dtype_u(init)`` hands back the dtype the problem was built with, so a
genuine cascade needs the construction dtype to become a per-sweep choice rather than a per-problem
one. It is the only part of this project a real dtype has made harder rather than easier.

**None of this is a statement about speed.** Everything is either emulated or genuinely stored at a
reduced dtype with backend-precision arithmetic. What a half-precision coarse level would actually
cost in time needs hardware that carries the format.
