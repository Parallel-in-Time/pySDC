Step-7: pySDC with external libraries
=====================================

pySDC can be used with external libraries, in particular for spatial discretization, parallelization and solving of linear and/or nonlinear systems.
In the following, we show a few examples of pySDC + X.

Part A: pySDC and FEniCS
------------------------


Part B: mpi4py-fft for parallel Fourier transforms
--------------------------------------------------


Part C: Time-parallel pySDC with space-parallel PETSc
-----------------------------------------------------

With rather unfavorable scaling properties, parallel-in-time methods are only really useful when spatial parallelization is maxed out.
To work with spatial parallelization, this part shows how to (1) include and work with an external library and (2) set up space- and time-parallel runs.
We use again the forced heat equation as our testbed and PETSc for the space-parallel data structures and linear solver.
See `implementations/datatype_classes/petsc_dmda_grid.py` and `implementations/problem_classes/HeatEquation_2D_PETSc_forced.py` for details on the PETSc bindings.

Important things to note:

- We need processors in space and time, which can be achieved by `comm.Split` and coloring. The space-communicator is then passed to the problem class.
- Below we run the code 3 times: with 1 and 2 processors in space as well as 4 processors (2 in time and 2 in space). Do not expect scaling due to the CI environment.

.. include:: doc_step_7_C.rst
