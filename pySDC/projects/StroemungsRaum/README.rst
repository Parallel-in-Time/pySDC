StroemungsRaum
==============

**StroemungsRaum** is a research software project developed within the
BMBF-funded project

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

Funding
-------
Funded by the **German Federal Ministry of Education and Research (BMBF)** under
grant number **16ME0708**.

