Finite elements with pySDC: the mass-matrix route
=================================================

This project shows how to combine pySDC with finite elements, using `Firedrake <https://www.firedrakeproject.org>`_
for the spatial discretisation. It runs SDC, its multilevel variant MLSDC, and the parallel-in-time PFASST on four
problems, and its main finding is this: **with high-order elements, a multilevel hierarchy pays; with linear
elements, it does not.** At the same number of unknowns, three-level MLSDC reaches only 0.65x-0.84x the speed of SDC
with linear elements, i.e. it costs more, but is 1.11x-1.94x faster than SDC with fourth-order elements.

A finite element discretisation of :math:`u_t = f(u)` gives a system :math:`M u' = R(u)`, where :math:`u` holds the
coefficients of the solution, :math:`M` is the *mass matrix* and :math:`R(u)` is the assembled right-hand side. Both
:math:`M u` and :math:`R(u)` are *load vectors*: their entries are integrals against the test functions, so they live
in the dual space of the finite element space, not in the space itself. The usual way to bring this into a time
integrator is to multiply by :math:`M^{-1}`, which costs a linear solve per right-hand side evaluation. This project
takes the other route throughout: **the mass matrix is never inverted, anywhere.** The problem classes hand the
sweepers load vectors, and the mass-matrix sweepers (``generic_implicit_mass``, ``imex_1st_order_mass``) and the
mass-matrix transfer between levels (``base_transfer_mass``) work with those directly.

The 1d problems come with continuous (``CG``, the usual Lagrange elements) and discontinuous (``DG``) elements, and
each multilevel hierarchy can be coarsened in either of two directions: a coarser mesh at fixed element order
(*h-coarsening*), or a lower element order on the same mesh (*p-coarsening*). Each example declares the combinations
it has a problem class for; ``get_families`` and ``get_coarsenings`` in ``setups_firedrake.py`` report them.

What is here
------------

``setups_firedrake.py``
    Descriptions for the four examples:

    - ``heat`` -- forced heat equation in 1D, IMEX: diffusion implicit, forcing explicit.
    - ``burgers`` -- viscous Burgers in 1D, fully implicit, with a Newton solve per collocation node.
    - ``grayscott`` -- Gray-Scott reaction-diffusion in 1D, fully implicit, with a Newton solve per collocation node.
    - ``vortex`` -- vorticity-velocity in 2D, periodic, IMEX. The coarse level does not pay for itself here (0.79x
      on two levels); it is in the study as the 2d and periodic case, not as a savings case. CG and h only.

    ``get_description(example, nlevels, family, coarsening)`` builds the pySDC description of any declared
    combination.

``problem_classes/Burgers_1D_Firedrake.py``
    The CG Burgers problem. The CG heat, Gray-Scott and vortex problems are general enough to live in
    ``pySDC/implementations/problem_classes`` (``HeatFiredrake``, ``GrayScott_1D_Firedrake``,
    ``VorticityVelocity_2D_Firedrake_periodic``).

``problem_classes/DG_1D_Firedrake.py``
    The DG versions of the three 1d problems: symmetric interior penalty (SIPG) for diffusion, Nitsche terms for
    Dirichlet data, and a local Lax-Friedrichs flux for the Burgers advection.

    The project's problem classes are built on ``FormFiredrake``: they give the mass form and the right-hand side
    form, and ``FormFiredrake`` assembles them, solves the implicit systems and applies the mass matrix.

``run_examples_firedrake.py``
    Runs every declared combination with SDC, MLSDC on 2 and 3 levels, and PFASST on up to 8 parallel steps, then the
    element-order comparison. Every table below comes from it.

``pySDC/tests/test_projects/test_FEM_with_Firedrake/``
    The tests, which assert the claims below so they stay true. They are marked ``firedrake`` and run in pySDC's
    Firedrake CI jobs.

How to run it
-------------

You need Firedrake, which needs Python 3.11 or newer and NumPy 2.1 or newer. Firedrake's own instructions are at
https://www.firedrakeproject.org/install.html. pySDC's CI installs it on top of conda-forge's PETSc, from the root of
the pySDC repository:

.. code-block:: bash

    micromamba env create -f etc/environment-firedrake.yml
    micromamba activate pySDC
    export PETSC_DIR=$CONDA_PREFIX PETSC_ARCH= CC=mpicc CXX=mpicxx
    # Linux with conda-forge's compilers only: without these, every compiled kernel fails
    export PYOP2_CFLAGS='-fPIC -std=gnu11 -O3' PYOP2_LDFLAGS='-shared'
    pip install libsupermesh firedrake-rtree petsctools
    pip install --no-build-isolation --no-deps firedrake==2026.10.0
    pip install firedrake==2026.10.0
    pip install --no-deps -e .

The PETSc version pinned in ``etc/environment-firedrake.yml`` must match the Firedrake release; the comments there and
in the ``user_firedrake_tests_conda`` job of ``.github/workflows/ci_pipeline.yml`` explain the steps. Then, from the
directory where the output should go:

.. code-block:: bash

    python -m pySDC.projects.FEM_with_Firedrake.run_examples_firedrake

This writes every table below to ``data/fem_with_firedrake_out.txt`` and the element-order study as a plot to
``data/fem_with_firedrake_orders.png``. The settings are sized for CI runtime, not for a production run. The tests
run with

.. code-block:: bash

    pytest -m firedrake pySDC/tests/test_projects/test_FEM_with_Firedrake

To build your own setup, copy one from ``setups_firedrake.py``. The three pieces that matter are:

.. code-block:: python

    description['sweeper_class'] = generic_implicit_mass      # or imex_1st_order_mass
    description['base_transfer_class'] = base_transfer_mass   # tau by P^T, u0 and u by L2 projection
    description['base_transfer_params'] = {'finter': False}

and a problem class whose ``eval_f`` returns the assembled weak form :math:`R(u)` rather than :math:`M^{-1}R(u)`, whose
``solve_system`` takes a right-hand side that is already a load vector, and which implements ``apply_mass_matrix``.
``FormFiredrake`` does all three, given the forms.

Then: **high order, coarsened in h.** CG or DG, they perform the same. The rest of this file is why, and what it took
to make the DG half true.

How the work is counted
~~~~~~~~~~~~~~~~~~~~~~~

The tables compare *work*, not wall-clock time: the mean number of iterations per time step times the cost of one
iteration in fine-level sweeps. One iteration costs one sweep on every level, each charged by its number of degrees
of freedom (dofs) relative to the fine level, plus the restrictions of the solution to the coarse levels, each charged
as 1% of a fine sweep (``RESTRICTION_COST`` in ``run_examples_firedrake.py``). The *speed-up* is the work of SDC
divided by the work of the configuration; below 1.00x means MLSDC costs more than SDC.

Residual tolerances
~~~~~~~~~~~~~~~~~~~

The iterations stop when the residual of the collocation problem drops below ``restol``. That residual is a load
vector, and pySDC's Firedrake data types measure load vectors in the *dual norm*, the norm dual to :math:`L^2`:
:math:`\sqrt{r^T M^{-1} r}`, which is the :math:`L^2` norm of the function :math:`w` with :math:`(w, v) = r(v)` for
every test function :math:`v`. Unlike a norm of the vector entries, it does not shrink as the mesh is refined, so
one tolerance per example serves every element family, order and level: 2e-9 for ``heat``, 1e-6 for ``burgers``, 1e-8
for ``grayscott`` and 1e-7 for ``vortex`` (``SETUPS`` in ``setups_firedrake.py``).

Use high-order elements
-----------------------

This is not a detail: it is what makes the multilevel hierarchy pay at all. At *identical* dof counts on every level,
with only the element order changed (and the mesh refined to compensate), on h-coarsened hierarchies:

=============  =============  =============  =============
example        CG1            CG2            CG4
=============  =============  =============  =============
``heat``       0.93x / 0.84x  1.24x / 1.05x  1.24x / 1.57x
``burgers``    0.83x / 0.73x  0.87x / 0.73x  1.30x / 1.94x
``grayscott``  0.77x / 0.65x  0.83x / 0.70x  1.14x / 1.11x
=============  =============  =============  =============

(speed-up of MLSDC over SDC on 2 / 3 levels)

At CG1 nothing pays: every example loses on two and on three levels. At CG4 every example pays on both. The coarse
level exists to approximate the smooth part of the error, and for smooth solutions a high-order space on a coarser mesh
does that far better than a low-order space on a finer one at the same cost.

``test_examples_firedrake.py::test_high_order_elements_pay`` asserts this in CI and plots the three-level speed-ups
to ``data/fem_with_firedrake_orders.png``, the picture on this project's card in the gallery. To stay within CI
runtime it runs 2 time steps instead of 8 and only CG1 and CG4, so the plotted values are close to, but not the same
as, the ones in the table.

This is the finite-element form of a result already known for finite differences, where MLSDC needs high-order
*interpolation* -- orders 4 to 6, with restriction left at order 2 (see ``pySDC/tutorial/step_4``). The natural
prolongation between nested finite element spaces **is** the inclusion, and its approximation order is the element
order. So "use CG4" here and "use ``iorder=6``" there are the same statement about the same operator.

Coarsen in h, not in p
----------------------

The same statement, pointed at the coarsening direction. Both ladders are nested, and with CG both halve the dof count
per level -- ``CG4`` on meshes of 512/256/128 cells against ``CG4/CG2/CG1`` on 512 cells give the identical
2049/1025/513 -- so for CG the two cost exactly the same per iteration and only the quality of the coarse space
differs. (With DG the p ladder cannot halve: order 4/2/1 means 5/3/2 dofs per cell, so p-coarsening is charged more
there as well.)

==============  ==========  ==============  ==============
config          SDC (work)  MLSDC 2 levels  MLSDC 3 levels
==============  ==========  ==============  ==============
heat CG h       5.75        4.62 (1.24x)    3.66 (1.57x)
heat CG p       5.75        4.62 (1.24x)    5.49 (1.05x)
heat DG h       5.75        4.62 (1.24x)    3.66 (1.57x)
heat DG p       5.75        4.92 (1.17x)    6.24 (0.92x)
burgers CG h    4.00        3.08 (1.30x)    2.06 (1.94x)
burgers CG p    4.00        4.62 (0.87x)    5.49 (0.73x)
burgers DG h    4.00        3.08 (1.30x)    2.06 (1.94x)
burgers DG p    4.00        4.92 (0.81x)    6.24 (0.64x)
grayscott CG h  6.38        5.58 (1.14x)    5.72 (1.11x)
grayscott CG p  6.38        7.70 (0.83x)    9.16 (0.70x)
grayscott DG h  6.38        5.58 (1.14x)    5.72 (1.11x)
grayscott DG p  6.38        8.20 (0.78x)    10.40 (0.61x)
vortex CG h     5.38        6.77 (0.79x)    7.48 (0.72x)
==============  ==========  ==============  ==============

(work, with the speed-up over SDC in brackets)

h-coarsening is at least as good as p-coarsening in all six cases, and on three levels strictly better in all six.
The reason is the one above: dropping from ``CG4`` to ``CG2`` to ``CG1`` on a fixed mesh leaves a coarse space with
:math:`O(h^2)` approximation error, while keeping ``CG4`` and doubling the cell size gives :math:`O((2h)^5)`. The coarse
level is there to resolve the smooth part of the error, and a low order resolves it badly however fine the mesh.

With p-coarsening, the third level (order 1) lowers the iteration count in none of the six cases, so it only adds
cost: three levels are worse than two every time.
Order 2 still approximates the smooth error; order 1 does not.

CG or DG
--------

The same, once the DG hierarchy is built correctly. Of the 42 pairs of CG and DG iteration counts in the MLSDC table
above and the PFASST table below, 39 agree exactly, all of the MLSDC ones among them. The other three are PFASST runs
(heat with p-coarsening on 4 and 8 steps, Gray-Scott with h-coarsening on 2 steps) and differ by 0.12 iterations per
step, i.e. one iteration summed over the 8 steps, which is where the iteration happens to cross its tolerance.

That took two fixes, both of which CG gets for free and neither of which shows up in a discretisation test. Both are
in the defect list below, items 5 and 6. Before them, DG looked like a dead end. The "broken hierarchy" column was
measured in the original implementation of this study, before the fixes; the other two are the present ones:

=========================  ================  ==================  ================
DG, h-coarsening           broken hierarchy  Galerkin hierarchy  CG for reference
=========================  ================  ==================  ================
heat, 3 levels             1.10x             1.57x               1.57x
burgers, 3 levels          0.79x             1.94x               1.94x
grayscott, 3 levels        0.65x             1.11x               1.11x
heat, PFASST 8 steps       12.88 iters       4.50 iters          4.50 iters
burgers, PFASST 8 steps    13.50 iters       4.00 iters          4.00 iters
grayscott, PFASST 8 steps  did not converge  5.38 iters          5.38 iters
=========================  ================  ==================  ================

The difference that remains: at the same mesh and order, DG carries 25% more dofs (5 per cell against 4) for the same
accuracy, because these solutions are smooth. Same iterations, more work per iteration. DG earns those dofs on
discontinuities and on advection-dominated transport, which none of these examples have -- so use it here only if you
want it for other reasons, and know that the multilevel machinery will not hold you back.

The penalty constant :math:`\sigma` of the interior penalty method is not a tuning knob. In the original
implementation, sweeping it over 1, 2, 5, 10, 40 changed nothing above the coercivity threshold (the value below which
the discrete operator stops being positive definite); only :math:`\sigma = 1` sat below it and wrecked the coarse
correction. The setups use :math:`\sigma = 10`. What matters is not how big it is but that it is *the same on every
level*, which is item 6.

What it demonstrates
--------------------

**Savings** -- see the tables above. With h-coarsening, a third level pays clearly on heat and Burgers and is about
neutral on Gray-Scott. Expect less from a multilevel hierarchy the more nonlinear the problem is.

**Stability** -- PFASST with two levels, mean iterations per step as parallel steps are added:

==============  ======  ====  ====  =====
config          1 step  2     4     8
==============  ======  ====  ====  =====
heat CG h       3.00    3.38  3.75  4.50
heat CG p       3.00    3.50  3.88  4.62
heat DG h       3.00    3.38  3.75  4.50
heat DG p       3.00    3.50  4.00  4.75
burgers CG h    2.00    2.50  3.12  4.00
burgers CG p    3.00    3.00  3.38  4.12
burgers DG h    2.00    2.50  3.12  4.00
burgers DG p    3.00    3.00  3.38  4.12
grayscott CG h  3.62    3.75  4.50  5.38
grayscott CG p  5.00    5.12  5.38  5.75
grayscott DG h  3.62    3.88  4.50  5.38
grayscott DG p  5.00    5.12  5.38  5.75
vortex CG h     5.25    6.25  7.75  10.38
==============  ======  ====  ====  =====

The growth out to 8 parallel steps is 1.15-2.0x, which is what PFASST is supposed to do. Every PFASST solution agrees
with the one-step run in the :math:`L^2` norm: to 1.9e-8 or better for heat and Gray-Scott, 1.9e-7 for the vortex and
3.7e-7 for Burgers.

PFASST is the direction that is sensitive to how the solution is restricted to the coarse level: ``base_transfer_mass``
restricts ``u`` by :math:`L^2` projection rather than by sampling it at the coarse nodes. The MLSDC table cannot tell
the two apart, but in the original implementation the projection was worth up to a factor of two in PFASST iterations,
and on the vortex at 8 steps it made the difference between a converged answer and an :math:`O(1)` wrong one.

Why earlier attempts did not pay off
------------------------------------

Multilevel SDC with finite elements had been tried in pySDC before and did not pay. It took fixing six separate
defects. Note what they have in common: every one of them leaves a method that still converges, still to the right
answer, with a coarse level that corrects far less than it should. None of them is visible in a discretisation test,
and none of them raises anything. Here :math:`P` is the prolongation from a coarse to a fine level and :math:`\tau` is
the FAS correction, the term that makes the coarse problem consistent with the fine one.

1. **The FAS** :math:`\tau` **was restricted by interpolation.** :math:`\tau` is a load vector, not a nodal function,
   so it has to be restricted with :math:`P^T`. Interpolating it is wrong by roughly :math:`2^d` in :math:`d`
   dimensions.
2. **The dual convention stopped at the first coarse level.** ``u0`` was carried in the dual space (as :math:`M u_0`)
   on coarse levels, but only the finest transfer put it there, so three levels did not converge.
3. **The same convention broke across step boundaries**, because ``uend`` was handed over as a function rather than
   a load vector. That is why PFASST did not work with the mass matrix. Both are gone with the convention itself:
   ``u0`` is now a solution on every level, restricted by :math:`L^2` projection like the nodes, which hands the
   coarse sweep the same :math:`P^T M u_0`.
4. **The coarsening hid all of it.** Reducing the collocation nodes to one on the coarse level makes it
   asymptotically inert -- it can neither help nor hurt, and it masks a broken transfer. Combined with
   ``Problem.apply_mass_matrix`` silently defaulting to the identity (it now raises), a wrong mass matrix produced a
   stalled iteration many sweeps later rather than an error.
5. **The prolongation was not the inclusion, for DG.** Between nested spaces, :math:`P` should be the inclusion: a
   coarse function, unchanged, seen as a fine one. The original transfer built it by interpolation across meshes, i.e.
   by evaluating the coarse function at the fine dof positions. A fine dof sitting *on* a coarse facet has two coarse
   values there, and the point evaluation returns whichever cell its search finds first. The prolonged function is
   therefore continuous at every coarse facet: the jumps, which are the whole point of a DG space, are deleted. The
   error is :math:`O(1)` in the jump and exactly zero for smooth data, which is why it survived a convergence test at
   :math:`O(h^{p+1})`. A related trap on periodic meshes: dof coordinates taken from a global list place a dof on the
   periodic boundary on the far side of the domain, so the coarse function is extrapolated there. Here the
   prolongation is Firedrake's own (``MeshToMeshFiredrakeHierarchy`` for h, interpolation on the same mesh,
   ``MeshToMeshFiredrake``, for p), which is the inclusion; ``test_prolongation_preserves_jumps`` and
   ``test_prolongation_on_a_periodic_space`` check it with random coarse data.
6. **The interior penalty was rediscretised on every level.** The CG bilinear form does not know which mesh it lives
   on, so rediscretising it on a coarse level gives exactly the Galerkin operator :math:`P^T A_F P`. The SIPG form
   does know: its penalty scales as :math:`\sigma p^2 / h`, so a coarser mesh halves it and a lower order divides it
   by :math:`(p_f/p_c)^2`. That is not a small perturbation -- the penalty outweighs the volume term by
   :math:`\sigma p^2`, so the coarse operator was wrong in its *dominant* term and corrected almost nothing.
   ``setups_firedrake.py`` therefore pins the penalty coefficient to :math:`\alpha_l = \sigma p_0^2 \, h_l / h_0` on
   level :math:`l` (level 0 is the finest), so that :math:`\alpha_l / h_l` is the same on every level. With that and
   item 5, the rediscretised coarse operator is the Galerkin one, for both coarsening directions.

The collocation nodes are kept on every level for a further reason: partial node coarsening (5 -> 4, 5 -> 3) was worse
than no coarse level at all in the original implementation, because it destroys the stiff-limit annihilation that the
``LU`` preconditioner provides.

Stopping at the spatial error
-----------------------------

A residual tolerance is a guess at how accurate the time stepping needs to be. Below the error of the spatial
discretization, further iterations cannot make the solution more accurate. ``get_description(...,
spatial_error_floor=True)`` adds the convergence controller
:class:`~pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake.SpatialErrorFloorFiredrake`,
which also stops a step once its residual is below its share :math:`e_x \Delta t / T` of the spatial error :math:`e_x`.
It estimates :math:`e_x` once per step from what the next coarser mesh of the hierarchy cannot represent. The study
runs it for SDC and 2-level MLSDC (CG, h), iterations per step, 8 steps:

=============  =====================  =======================  ===============  =============
example        SDC: restol / + floor  MLSDC: restol / + floor  change in u      spatial error
=============  =====================  =======================  ===============  =============
``heat``       5.75 / 5.75            3.00 / 3.00              0                1.3e-15
``burgers``    4.00 / 4.00            2.00 / 2.00              0                5.6e-11
``grayscott``  6.38 / 6.38            3.62 / 3.62              0                4.6e-08
``vortex``     5.38 / 1.25            5.25 / 1.12              1.6e-2 / 6.9e-3  9.9e-02
=============  =====================  =======================  ===============  =============

The three 1D problems are resolved in space far beyond their residual tolerance, so the floor never stops a step
early. The vortex is not: its spatial error is about 0.1, and one sweep per step already gets the time stepping below
its share of it. Stopping there saves four of every five iterations, and the solution moves by much less than its
spatial error.

Earlier versions of this study
------------------------------

The study was first done with FEniCS, which pySDC no longer supports, and moved to Firedrake with the same problems
and settings. The findings marked "in the original implementation" above come from that version and were not
repeated; the conclusions of both versions are the same.
