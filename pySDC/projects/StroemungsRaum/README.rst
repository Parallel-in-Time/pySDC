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
Implemented examples and test cases include:

- Heat equation
- Convection–diffusion and nonlinear convection–diffusion problems
- Incompressible Navier–Stokes equations, using:
   - Projection methods
   - Monolithic formulations
   - DAE- and PDE sweepers

These serve as benchmarks and demonstrators for scalable space–time CFD
simulations.

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

Funding
-------
Funded by the **German Federal Ministry of Research, Technology and Space (BMFTR)**
under grant number **16ME0708**.

