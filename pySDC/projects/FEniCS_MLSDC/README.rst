Finite elements with pySDC: the mass-matrix route
=================================================

This is the reference for combining pySDC with finite elements. It shows SDC, MLSDC and PFASST on
three FEniCS problems, using the **mass-matrix formulation throughout** -- the mass matrix is never
inverted, anywhere.

Each problem comes with continuous (``CG``) and discontinuous (``DG``) elements, and each hierarchy
can be coarsened in either direction: a coarser mesh at fixed element order (**h**), or a lower
element order on the same mesh (**p**).

What is here
------------

``setups.py``
    Descriptions for the three examples. One linear, two nonlinear:

    - ``heat`` -- forced heat equation in 1D, IMEX.
    - ``burgers`` -- viscous Burgers in 1D, fully implicit with a node-local Newton.
    - ``grayscott`` -- Gray-Scott reaction-diffusion in 1D, fully implicit with a node-local Newton.

    ``get_description(example, nlevels, family, coarsening)`` builds any of the twelve combinations.

``problem_classes/DG_1D_FEniCS.py``
    The DG counterparts of all three: interior penalty for diffusion, Nitsche for Dirichlet data,
    a Lax-Friedrichs flux for the Burgers advection. Each is its CG parent with the weak form
    replaced -- the Newton loop, the mass matrix and the solver interface are inherited unchanged.

``run_examples.py``
    Runs every example, family and coarsening direction with SDC, MLSDC on 2 and 3 levels, and
    PFASST on up to 8 parallel steps. Takes about four minutes.

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

Then: **high order, coarsened in h.** CG or DG, they perform identically. The rest of this file is
why, and what it took to make the DG half true.

Use high-order elements
-----------------------

This is not a detail: it is what makes the multilevel hierarchy pay at all. At *identical*
fine-level dof counts, with only the element order changed:

=============  =============  =============  =============
example        CG1            CG2            CG4
=============  =============  =============  =============
``heat``       0.92x / 1.05x  1.22x / 1.05x  1.22x / 1.57x
``burgers``    0.89x / 0.76x  0.94x / 0.81x  1.26x / 1.94x
``grayscott``  0.87x / 0.74x  0.87x / 0.74x  1.24x / 1.24x
=============  =============  =============  =============

(speed-up at 2 / 3 levels; below 1.00x means MLSDC costs more than SDC)

At CG1 every example *loses*. The coarse level exists to approximate the smooth part of the error,
and for smooth solutions a high-order space on a coarser mesh does that far better than a low-order
space on a finer one at the same cost.

This is the finite-element form of a result already known for finite differences, where MLSDC needs
high-order *interpolation* -- orders 4 to 6, with restriction left at order 2 (see ``tutorial/step_4``).
The natural prolongation between nested finite element spaces **is** the inclusion, and its
approximation order is the element order. So "use CG4" here and "use ``iorder=6``" there are the same
statement about the same operator.

Coarsen in h, not in p
----------------------

The same statement, pointed at the coarsening direction. Both ladders are nested, and with CG both
halve the dof count per level -- ``CG4`` on meshes of 512/256/128 cells against ``CG4/CG2/CG1`` on
512 cells give the identical 2049/1025/513 -- so for CG the two cost exactly the same per iteration
and only the quality of the coarse space differs. (With DG the p ladder cannot halve: order 4/2/1
means 5/3/2 dofs per cell, so p-coarsening is charged more there as well.)

==============  ==========  ==============  ==============
config          SDC (work)  MLSDC 2 levels  MLSDC 3 levels
==============  ==========  ==============  ==============
heat CG h       5.75        4.50 (1.28x)    3.50 (1.64x)
heat DG h       5.75        4.50 (1.28x)    3.50 (1.64x)
heat CG p       5.75        4.50 (1.28x)    5.25 (1.09x)
heat DG p       5.75        4.80 (1.20x)    6.00 (0.96x)
burgers CG h    4.12        3.19 (1.29x)    1.97 (2.09x)
burgers DG h    4.12        3.19 (1.29x)    1.97 (2.10x)
burgers CG p    4.12        4.50 (0.92x)    5.25 (0.79x)
burgers DG p    4.12        4.80 (0.86x)    6.00 (0.69x)
grayscott CG h  6.50        5.44 (1.20x)    5.47 (1.19x)
grayscott DG h  6.50        5.44 (1.20x)    5.47 (1.19x)
grayscott CG p  6.50        7.50 (0.87x)    8.76 (0.74x)
grayscott DG p  6.50        8.00 (0.81x)    10.00 (0.65x)
==============  ==========  ==============  ==============

(work = iterations x summed dof ratio, so every level is charged for what it costs)

h-coarsening wins in every one of the six cases. The reason is the one above: dropping from ``CG4``
to ``CG2`` to ``CG1`` on a fixed mesh leaves a coarse space with :math:`O(h^2)` approximation error,
while keeping ``CG4`` and doubling the cell size gives :math:`O((2h)^5)`. The coarse level is there
to resolve the smooth part of the error, and a low order resolves it badly however fine the mesh.

Note also that p-coarsening never gets past the second level: two levels are close to h-coarsening,
three are worse than two. ``CG2`` still approximates the smooth error; ``CG1`` does not.

CG or DG
--------

Identical, once the DG hierarchy is built correctly. Same iteration counts, same speed-ups, same
PFASST growth, on all three examples -- read the table above in pairs.

That took two fixes, both of which CG gets for free and neither of which shows up in a
discretisation test. Both are in the defect list below, items 5 and 6. Before them, DG looked like a
dead end:

=========================  ================  ==================  ================
DG, h-coarsening           broken hierarchy  Galerkin hierarchy  CG for reference
=========================  ================  ==================  ================
heat, 3 levels             1.10x             1.64x               1.64x
burgers, 3 levels          0.79x             2.10x               2.09x
grayscott, 3 levels        0.65x             1.19x               1.19x
heat, PFASST 8 steps       12.88 iters       4.62 iters          4.62 iters
burgers, PFASST 8 steps    13.50 iters       4.12 iters          4.12 iters
grayscott, PFASST 8 steps  did not converge  6.75 iters          6.00 iters
=========================  ================  ==================  ================

The one honest difference that remains: at the same mesh and order DG carries 25% more dofs (5 per
cell against 4) for the same accuracy, because these solutions are smooth. Same iterations, more
work per iteration. DG earns those dofs on discontinuities and on advection-dominated transport,
which none of these examples have -- so use it here only if you want it for other reasons, and know
that the multilevel machinery will not hold you back.

The penalty constant :math:`\sigma` is not a tuning knob. Sweeping it over 1, 2, 5, 10, 40 changes
nothing above the coercivity threshold; only :math:`\sigma = 1` sits below it and wrecks the coarse
correction. Set it just clear of the threshold and stop thinking about it. What matters is not how
big it is but that it is *the same on every level*, which is item 6.

What it demonstrates
--------------------

**Savings** -- see the table above. A third level pays clearly on the smooth problems and is neutral
on Gray-Scott. Expect less from a multilevel hierarchy the more nonlinear the problem is.

**Stability** -- PFASST iteration counts as parallel steps are added, every run agreeing with serial
to the tolerance of the example:

==============  ======  ====  ====  =====
config          1 step  2     4     8
==============  ======  ====  ====  =====
heat CG h       3.00    3.38  3.88  4.62
heat DG h       3.00    3.38  3.88  4.62
heat CG p       3.00    3.50  4.00  4.75
heat DG p       3.00    3.50  4.00  4.75
burgers CG h    2.12    2.62  3.25  4.12
burgers DG h    2.12    2.62  3.25  4.12
burgers CG p    3.00    3.00  3.38  4.12
burgers DG p    3.00    3.00  3.38  4.12
grayscott CG h  3.62    3.88  4.50  6.00
grayscott DG h  3.62    4.00  4.75  6.75
grayscott CG p  5.00    5.62  6.88  9.25
grayscott DG p  5.00    6.00  8.12  12.12
==============  ======  ====  ====  =====

Growth out to 8 parallel steps is 1.4-2.4x, which is what PFASST is supposed to do.

Why earlier attempts did not pay off
------------------------------------

Six separate defects, each of which quietly capped or broke the multilevel gain. Note what they have
in common: every one of them leaves a method that still converges, still to the right answer, with a
coarse level that corrects far less than it should. None of them is visible in a discretisation
test, and none of them raises anything.

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
5. **The prolongation was not the inclusion, for DG.** ``mesh_to_mesh_fenics`` built :math:`P` with
   ``df.interpolate``. Across meshes that is point evaluation, and a fine dof sitting *on* a coarse
   facet has two coarse values there -- dolfin returns whichever cell the bounding-box tree finds
   first. The prolonged function is therefore continuous at every coarse facet: the jumps, which are
   the whole point of a DG space, are deleted. The error is :math:`O(1)` in the jump and exactly
   zero for smooth data, which is why it survived a convergence test at
   :math:`O(h^{p+1})`. :math:`P` is now assembled cell by cell, evaluating the coarse basis in the
   coarse cell that *contains* each fine cell. For continuous spaces this reproduces the old
   construction to machine precision.
6. **The interior penalty was rediscretised on every level.** The CG bilinear form does not know
   which mesh it lives on, so rediscretising it on a coarse level gives exactly the Galerkin
   operator :math:`P^T A_F P`. The SIPG form does know: its penalty scales as
   :math:`\sigma p^2 / h`, so a coarser mesh halves it and a lower order divides it by
   :math:`(p_f/p_c)^2`. That is not a small perturbation -- the penalty outweighs the volume term by
   :math:`\sigma p^2`, so the coarse operator was wrong in its *dominant* term and corrected almost
   nothing. ``setups.py`` now pins :math:`\alpha_l = \sigma p_0^2 \, h_l / h_0` so that
   :math:`\alpha_l / h_l` is the same on every level. With that and item 5,
   :math:`A_G = P^T A_F P` holds to machine precision, for both coarsening directions.

Nodes are kept on every level here for a further reason: partial node coarsening (5 -> 4, 5 -> 3) is
worse than no coarse level at all, because it destroys the stiff-limit annihilation that the ``LU``
preconditioner provides.

Settings are sized for CI runtime, not for a production run. Reproduce with ``run_examples.py``.
