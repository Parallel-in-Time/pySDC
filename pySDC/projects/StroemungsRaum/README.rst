StroemungsRaum
==============

**StroemungsRaum** is a research software project developed within the
BMFTR-funded project

*“StrömungsRaum – Novel Exascale Architectures with Heterogeneous Hardware
Components for Computational Fluid Dynamics Simulations”*
(October 2022 – September 2025).

The project addresses the development of scalable numerical methods and
high-performance algorithms for Computational Fluid Dynamics (CFD) targeting
future **exascale computing architectures** with heterogeneous hardware.

Scope of This Repository
------------------------
This repository contains the **Forschungszentrum Jülich (FZJ)** contribution to
the StrömungsRaum project, focusing on:

- Parallel-in-time methods
- Combined space–time parallelization for fluid simulations
- Algorithmic scalability for time-dependent PDEs

The goal is to expose concurrency beyond spatial parallelism and enable
efficient execution on large-scale HPC systems.

Model Problems and Methods
--------------------------
The examples are finite-element discretizations in space, solved in time with
spectral deferred corrections (SDC), and serve as benchmarks and demonstrators
for scalable space–time CFD simulations:

- the forced heat equation on the unit square,
- the convection–diffusion of a rotating Gaussian,
- the incompressible Navier–Stokes equations for the flow around a cylinder,
  with a projection scheme and in monolithic form,
- the incompressible Navier–Stokes equations with a manufactured Taylor–Green
  solution, to study the order reduction caused by time-dependent boundary
  conditions.

All of them use `Firedrake <https://www.firedrakeproject.org>`_, a finite-element
library in which a problem is written as weak forms in UFL, its form language. The
problem classes are built on pySDC's generic
``pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`` (everything
implicit) and ``FormFiredrakeIMEX`` (with an explicit part), which assemble the
semi-discrete system :math:`M u' = R(u)` from a mass form and right-hand-side forms.
They never invert the mass matrix :math:`M`: the right-hand side is handed to the
sweepers as a *load vector*, the vector of the form's values on the basis functions,
which lives in the dual space of the finite-element functions. Accordingly, they run
with pySDC's mass-matrix sweepers ``generic_implicit_mass`` and
``imex_1st_order_mass``, and the SDC residual, itself a load vector, is measured in
the *dual norm*, the norm dual to the :math:`L^2` norm, which does not shrink with
the mesh width.

The project was first written with FEniCS, which pySDC no longer supports. The tests
check the Firedrake classes against values recorded with them and, for the cylinder
flow, against the FEATFLOW benchmark.

Organisation of the project
---------------------------
Problem classes, in ``problem_classes/``:

- ``HeatEquation_2D_Firedrake.py``: :math:`u_t = \nu \Delta u + f` on the unit square
  with the exact solution :math:`u = \sin(\pi x) \sin(\pi y) \cos(t) + c`, so the
  Dirichlet values depend on time. Diffusion is implicit, the forcing :math:`f`
  explicit (IMEX).
- ``ConvectionDiffusion_2D_Firedrake.py``: :math:`u_t = -U \cdot \nabla u + \nu \Delta u`
  on :math:`[-0.5, 0.5]^2` with the rotation :math:`U = (-4y, 4x)`, whose exact
  solution is a Gaussian that rotates and spreads. Diffusion is implicit, convection
  explicit.
- ``NavierStokes_2D_monolithic_Firedrake.py``: the flow around a cylinder of the
  *DFG 2D-3 benchmark* (Schäfer and Turek, 1996): a channel of size
  :math:`2.2 \times 0.41` with a cylinder of diameter 0.1, viscosity
  :math:`\nu = 10^{-3}` and an inflow that rises and falls again over
  :math:`t \in [0, 8]`. *Monolithic* means that velocity and pressure are solved for
  together, here with Taylor–Hood elements. The incompressibility constraint has no
  time derivative, so the mass matrix is singular and the system is a
  differential-algebraic equation; the mass-matrix sweeper handles this because it
  never inverts :math:`M`. The module also reads the mesh and computes the drag and
  lift coefficients of the cylinder, the quantities the benchmark compares.
- ``NavierStokes_2D_projection_Firedrake.py``: the same flow with a *projection
  scheme*: each implicit solve gives a tentative velocity, which a pressure Poisson
  equation and a velocity correction then project onto (approximately)
  divergence-free fields. The projection needs each node's distance from the start of
  the step, which the convergence controller ``convergence_controllers/prepare_step.py``
  (``PrepareStep``) hands to the problem once per step; with it, the scheme runs with
  pySDC's standard ``imex_1st_order_mass`` sweeper.
- ``NavierStokes_2D_TaylorGreen_monolithic_Firedrake.py``: monolithic Navier–Stokes
  with a manufactured, travelling Taylor–Green solution on :math:`[-0.5, 0.5]^2`, with
  either time-dependent Dirichlet or periodic boundaries in :math:`x`, and optionally
  differentiated boundary data, see below. Only the velocity is prescribed; the
  pressure is fixed up to its constant by a null space. Given a shared ``mesh`` (one
  for all steps, the meshes of a ``MeshHierarchy`` for the levels), it also runs with
  MLSDC and PFASST.

The mesh of the cylinder flow is ``meshs/cylinder.msh``, a Gmsh file whose inflow,
outflow, walls and cylinder are tagged, so the problem class loads it with
Firedrake's ``Mesh``.

Run scripts, in the project folder, and their plots, in ``plotting/``:

- ``run_heat_equation_Firedrake.py`` and ``run_convection_diffusion_equation_Firedrake.py``
  print the relative error at the final time and write the numerical and exact
  solution of every step to an HDF5 ``CheckpointFile`` of Firedrake in
  ``data/heat_equation`` or ``data/convection_diffusion``.
  ``plotting/plots_heat_equation_Firedrake.py`` and
  ``plotting/plots_convection_diffusion_equation_Firedrake.py`` animate them and save
  the last frame and contour plots as PNG files.
- ``run_accuracy/run_accuracy_heat_equation_Firedrake.py`` runs the heat equation with
  four step sizes and plots the errors: with two RADAU-RIGHT nodes, the observed order
  is the design order 3.
- ``run_Navier_Stokes_equations_Firedrake.py`` runs the projection scheme for the
  first steps of the cylinder flow and writes drag and lift of every step to
  ``data/navier_stokes/Lift_Drag_Coefficients.txt``, and velocity and pressure to a
  ``CheckpointFile`` there (hooks in ``hooks/hooks_NSE_IMEX_Firedrake.py``).
  ``plotting/plots_Navier_Stokes_projection_Firedrake.py`` draws velocity, pressure,
  vorticity, velocity magnitude and streamlines from it.
- ``run_Navier_Stokes_equations_monolithic_Firedrake.py`` runs the monolithic cylinder
  flow over the whole benchmark interval and records drag and lift of every step.
- ``run_Navier_Stokes_TaylorGreen_Firedrake.py`` runs the order study described below
  and prints errors and observed orders.

On the coarse mesh of the project, the drag and lift of both cylinder-flow
formulations after their first steps agree with the reference values of the FEATFLOW
benchmark solution to within the error this mesh allows: :math:`2 \cdot 10^{-2}` in
drag and :math:`10^{-4}` in lift for the monolithic run after one step of 0.01, and
:math:`5 \cdot 10^{-3}` and :math:`5 \cdot 10^{-5}` for the projection scheme at
:math:`t = 0.005`.

The run scripts advance one time step after the other (``controller_nonMPI`` with one
process). The problem classes take Firedrake's default communicator, so the mesh can
be distributed over several MPI ranks for parallelism in space; the tests check this
for the monolithic cylinder flow on two ranks.

Order reduction from time-dependent boundary conditions
-------------------------------------------------------
``run_Navier_Stokes_TaylorGreen_Firedrake.py`` runs a manufactured Taylor–Green
solution on :math:`[-0.5, 0.5]^2` that is exactly one-periodic in :math:`x` and
constant on the top and bottom boundary. The *same* solution can therefore be
computed with time-dependent Dirichlet conditions in :math:`x` or with periodic
ones, and the difference in the observed temporal order isolates the order
reduction caused by the time-dependent boundary data alone. Only the velocity is
prescribed; the pressure is fixed up to its constant by a null space.

The number of collocation nodes decides whether the effect is visible: RADAU-RIGHT
with :math:`M` nodes has design order :math:`2M-1`, and time-dependent boundary
data pull it down towards the stiff order. With :math:`M = 4` (24 × 24 squares,
:math:`\nu = 0.1`, :math:`\Delta t` from 0.2 to 0.025) the measured pressure orders
are 7.0 with periodic and 4.1 with time-dependent Dirichlet conditions, with finest
pressure errors of 2.6e-5 and 1.8e-2. The velocity loses less (6.9 and 5.7).

The third variant, ``differentiated_bc``, imposes the boundary data on its time
derivative and recovers the stage values by collocation quadrature instead of
evaluating the data pointwise at the nodes. This is the boundary-condition
analogue of the differentiated-constraint remedy explored in pull request #641,
and it removes most of the penalty: the pressure order goes up to 6.0 and its
finest error down to 1.3e-4, more than a hundred times smaller than with the
pointwise data and within a factor of five of the periodic case.

Installation
------------
The project needs Firedrake, which needs Python 3.11 or newer and NumPy 2.1 or
newer. See the `Firedrake installation instructions
<https://www.firedrakeproject.org/install.html>`_. pySDC's own way, used in its
continuous integration (the ``user_firedrake_tests_conda`` job in
``.github/workflows/ci_pipeline.yml``, which lists the exact commands), is to create
a conda environment from ``etc/environment-firedrake.yml``, which provides a prebuilt
PETSc, to pip-install Firedrake 2026.10.0 against it, and to install pySDC into that
environment with ``pip install -e .``. Then, from the root of the repository, for
example:

.. code-block:: bash

    python pySDC/projects/StroemungsRaum/run_heat_equation_Firedrake.py
    python pySDC/projects/StroemungsRaum/plotting/plots_heat_equation_Firedrake.py

Tests
-----
The tests are in ``pySDC/tests/test_projects/test_StroemungsRaum/`` and carry the
``firedrake`` marker; they run in pySDC's Firedrake CI jobs. One of them needs two
MPI ranks, which ``etc/run_mpi_tests.sh`` provides through ``mpi-pytest``:

.. code-block:: bash

    bash etc/run_mpi_tests.sh pySDC/tests/test_projects/test_StroemungsRaum firedrake

Funding
-------
Funded by the **German Federal Ministry of Research, Technology and Space (BMFTR)**
under grant number **16ME0708**.
