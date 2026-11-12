Low-precision implicit solves in SDC
====================================

In this project, we test whether the implicit solves of SDC can be done in reduced precision without
losing the accuracy of the collocation problem.
More precisely, we let the solver work on the correction of the iterate instead of the iterate itself
(the delta form), so that an error relative to the solver's result vanishes as the iteration converges.
Then, we measure and predict how many additional iterations a less accurate solve costs, and for which
problems this pays off.

Usage
-----

The delta form is part of pySDC.
``delta_implicit`` replaces ``generic_implicit`` and gives the same iterates:

.. code-block:: python

    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    description['sweeper_class'] = delta_implicit

If the implicit part is linear, set ``linear_implicit=True`` in the sweeper parameters, and the problem's
``solve_system`` receives the right-hand side of the correction equation.
If it is nonlinear, the problem class has to provide ``solve_system_delta``, which solves for the
correction directly.
Note that the increment :math:`f(u + \delta) - f(u)` has to be evaluated without cancellation, see
``allencahn_delta`` in ``problems.py`` for an example.
Both methods are documented in :class:`pySDC.core.problem.Problem`.

Organisation of the project
---------------------------

- ``problems.py`` contains the test problems in delta form: the heat equation, Allen-Cahn and the
  Van der Pol oscillator.
- ``inexact.py`` turns any problem class into one whose solves deliver only a given relative accuracy,
  or round their result to fp32 or fp16.
- ``genuine.py`` provides an LU decomposition which computes in fp32 or fp16 arithmetic.
- ``krylov.py`` provides conjugate gradients in fp64, fp32 or fp16, on a CPU or a GPU (there with fused
  kernels), and the periodic heat equation with CG node solves.
- ``multigrid.py`` provides the 3D periodic heat equation with geometric multigrid node solves
  (``pySDC.helpers.multigrid``) in fp64, fp32 or fp16, on a CPU or a GPU.
- ``precision_theory.py`` predicts how many additional iterations a less accurate solve costs.
- ``eta_spec.py`` collects the test cases and measures how accurate a solve has to be.
- ``work_precision.py`` computes the work needed to reach a given accuracy.

Tests
-----

All tests are in ``tests`` and run with

.. code-block:: bash

    pytest pySDC/projects/DeltaSDC/tests

The GPU tests are in ``pySDC/tests/test_projects/test_DeltaSDC`` and run with the other GPU tests of pySDC.

The results of an earlier, broader study with emulated low precision on the coarse levels of MLSDC,
storage precision raised during the iteration, ParaDiag and runs on GPUs are kept in the git tag
``deltasdc-precision-study-2026-11``.
