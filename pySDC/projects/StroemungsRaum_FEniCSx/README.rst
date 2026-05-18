StroemungsRaum_FEniCSx
======================

**StroemungsRaum_FEniCSx** is a research software project derived from the
original *StroemungsRaum* framework and developed within the context of the
BMBF-funded project

*“StrömungsRaum – Novel Exascale Architectures with Heterogeneous Hardware
Components for Computational Fluid Dynamics Simulations”*
(October 2022 – September 2025).

This repository provides a modernized implementation of the original
StroemungsRaum research codes using **FEniCSx** instead of the legacy FEniCS
framework. The migration enables compatibility with current finite element
software ecosystems and improved support for modern HPC architectures.

Scope of This Repository
------------------------
This repository contains the **Forschungszentrum Jülich (FZJ)** contribution to
the StrömungsRaum project implemented with FEniCSx, focusing on:

- Parallel-in-time methods
- Combined space–time parallelization
- Scalable solvers for time-dependent PDEs
- High-performance finite element implementations with PETSc and MPI
- Research on exascale-ready CFD algorithms

The primary objective is to expose concurrency beyond spatial parallelism and
enable efficient large-scale simulations on modern heterogeneous HPC systems.

Software Stack
--------------
This project is based on the modern FEniCSx ecosystem, including:

- FEniCSx
- PETSc
- mpi4py
- petsc4py
- UFL
- Basix
- DOLFINx

Research Focus
--------------
The software serves as a research platform for investigating:

- Scalable time integration methods
- Space–time parallelism
- High-order temporal discretizations
- Solver robustness for incompressible flows
- Parallel performance on heterogeneous architectures

Funding
-------
Funded by the **German Federal Ministry of Education and Research (BMBF)** under
grant number **16ME0708**.

