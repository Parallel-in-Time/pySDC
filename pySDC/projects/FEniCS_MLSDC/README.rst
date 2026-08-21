Finite elements with pySDC: the mass-matrix route
=================================================

This is the reference for combining pySDC with finite elements. It shows SDC, MLSDC and PFASST on
three FEniCS problems, using the **mass-matrix formulation throughout** -- the mass matrix is never
inverted, anywhere.

What is here
------------

``setups.py``
    Descriptions for the three examples. One linear, two nonlinear:

    - ``heat`` -- forced heat equation in 1D, IMEX.
    - ``grayscott`` -- Gray-Scott reaction-diffusion in 1D, fully implicit with a node-local Newton.
    - ``vortex`` -- vorticity-velocity formulation of the 2D Navier-Stokes equations, IMEX.

``run_examples.py``
    Runs each example with SDC, MLSDC on 2 and 3 levels, and PFASST on up to 8 parallel steps.

``tests/``
    Asserts the claims below, so they stay true.

How to run it
-------------

.. code-block:: bash

    micromamba env create -f pySDC/projects/FEniCS_MLSDC/environment.yml
    python pySDC/projects/FEniCS_MLSDC/run_examples.py

To build your own setup, copy one from ``setups.py``. The three pieces that matter are:

.. code-block:: python

    description['sweeper_class'] = generic_implicit_mass      # or imex_1st_order_mass
    description['base_transfer_class'] = base_transfer_mass   # restricts tau and u0 with P^T
    description['base_transfer_params'] = {'finter': False}

and a problem class whose ``eval_f`` returns the assembled weak form rather than
:math:`M^{-1}F`, whose ``solve_system`` takes a right-hand side that is already in the dual space,
and which implements ``apply_mass_matrix``.

Coarsen by **mesh refinement** and keep the collocation nodes on every level.

Why earlier attempts did not pay off
------------------------------------

Four separate defects, each of which quietly capped or broke the multilevel gain:

1. **The FAS** :math:`\tau` **was restricted by interpolation.** :math:`\tau` is a load vector, not a
   nodal function, so it has to be restricted with :math:`P^T`. Interpolating it is wrong by roughly
   :math:`2^d`.
2. **The dual convention stopped at the first coarse level.** ``u0`` is carried in the dual space on
   coarse levels, but only the finest transfer put it there, so three levels did not converge.
3. **The same convention broke across step boundaries**, because ``uend`` was handed over primal.
   That is why PFASST did not work with the mass matrix.
4. **The coarsening hid all of it.** Reducing the collocation nodes to one makes the coarse level
   asymptotically inert -- it can neither help nor hurt, and it masks a broken transfer. Combined
   with ``Problem.apply_mass_matrix`` silently defaulting to the identity, a wrong mass matrix
   produced a stalled iteration many sweeps later rather than an error.

Nodes are kept on every level here for a further reason: partial node coarsening (5 -> 4, 5 -> 3) is
worse than no coarse level at all, because it destroys the stiff-limit annihilation that the ``LU``
preconditioner provides.

What it demonstrates
--------------------

**Savings** (work = iterations x summed dof ratio, so a coarse level is charged for what it costs):

=============  ==========  ==========  ==========
example        SDC         MLSDC (2)   MLSDC (3)
=============  ==========  ==========  ==========
``heat``       5.75        3.00        2.00
speed-up       1.00x       1.28x       **1.64x**
``grayscott``  6.50        3.62        3.12
speed-up       1.00x       1.20x       1.19x
``vortex``     5.75        6.12        6.12
speed-up       1.00x       0.75x       0.72x
=============  ==========  ==========  ==========

A third level pays on the linear problem, is neutral on Gray-Scott, and **the vortex shows no gain at
all** -- it needs slightly *more* iterations with a coarse level than without. That is reported rather
than tuned away: it survives changing the preconditioner and the width of the shear layer (``delta``
0.05, 0.15 and 0.3 give identical counts), so the coarse level simply has nothing to contribute for
this problem at this size. The vortex is here as the *correctness* example -- a 2D nonlinear IMEX
problem giving the same answer under SDC, MLSDC and PFASST -- not as a savings example. Expect less
from a multilevel hierarchy the more nonlinear the problem is.

**Stability** -- PFASST iteration counts as parallel steps are added, every run agreeing with serial:

=============  ======  ======  ======  ======
example        1 step  2       4       8
=============  ======  ======  ======  ======
``heat``       3.00    3.38    3.88    4.62
``grayscott``  3.62    3.88    4.50    6.00
``vortex``     6.12    9.00    14.75   --
=============  ======  ======  ======  ======

The vortex scales visibly worse, and **beyond 4 steps it fails silently**: at 8 it reports convergence
after 22 iterations and returns a solution that is wrong by O(1). ``pfasst_procs`` in ``setups.py``
therefore stops at 4 for that example. Treat this as a limit of the example, not a green light --
whatever breaks there has not been diagnosed.

Settings are sized for CI runtime, not for a production run. Reproduce with ``run_examples.py``.
