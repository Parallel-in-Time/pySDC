Low-precision implicit solves in SDC
====================================

In this project, we test whether the implicit solves of SDC can be done in reduced precision without
losing the accuracy of the collocation problem.
More precisely, we let the solver work on the correction of the iterate instead of the iterate itself
(the delta form), so that an error relative to the solver's result vanishes as the iteration converges.
Then, we bound the errors of the solver that SDC tolerates, and measure for which solvers and problems this pays
off on GPUs.

Usage
-----

The delta form is part of pySDC: ``delta_implicit`` replaces ``generic_implicit`` and gives the same iterates.
With ``linear_implicit=True`` in the sweeper parameters, the problem's ``solve_system`` receives the right-hand
side of the correction equation, and may return its result in a lower precision:

.. code-block:: python

    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    description['sweeper_class'] = delta_implicit
    description['sweeper_params']['linear_implicit'] = True

Organisation of the project
---------------------------

The node solvers in fp64, fp32 or fp16:

- ``genuine.py`` provides an LU decomposition which computes in fp32 or fp16 arithmetic, and wraps a problem
  class with a matrix so that its node solves use it.
- ``krylov.py`` provides conjugate gradients in fp64, fp32 or fp16, on a CPU or a GPU (there with fused
  kernels), and the periodic heat equation with CG node solves.
- ``multigrid.py`` provides the 3D periodic heat equation with geometric multigrid node solves
  (``pySDC.helpers.multigrid``) in fp64, fp32 or fp16, on a CPU or a GPU.

FFT node solves in fp32 and fp16 are part of pySDC's finite-difference problem classes (``solve_precision``).

Tests
-----

All tests are in ``tests`` and run with

.. code-block:: bash

    pytest pySDC/projects/DeltaSDC/tests

The GPU tests are in ``pySDC/tests/test_projects/test_DeltaSDC`` and run with the other GPU tests of pySDC.

An earlier, broader study is kept in the git tag ``deltasdc-precision-study-2026-11``: emulated low precision on
the coarse levels of MLSDC, storage precision raised during the iteration, ParaDiag, a probabilistic prediction of
the additional iterations, format schedules per iteration, and further test problems such as Van der Pol.
