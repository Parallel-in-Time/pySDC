Step-7: pySDC with external libraries
=====================================

pySDC can be used with external libraries, in particular for spatial discretization, parallelization and solving of linear and/or nonlinear systems.
In the following, we show a few examples of pySDC + X.

Part A: pySDC and FEniCS
------------------------

In this example, pySDC is coupled with the FEniCS framework foir using finite elements in space.
This implies significant changes to the algorithm, depending on whether or not the mass matrix should be inverted.
SDC, MLSDC and PFASST can be used without changes when the right-hand side of the ODE is defined with the inverse of the mass matrix.
Otherwise, the mass matrix has to be used for e.g. the tau-correction.
This example tests different variants of this methodology for SDC, MLSDC and PFASST.

Important things to note:

- This example shows that even core routines like the `BaseTransfer` can be overwritten if needed.
- It is also valuable to check out the data type and transfer classes required to work with FEniCS. Both can be found in the `implementations` folder.

.. include:: doc_step_7_C.rst

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
