Delta-form SDC for mixed precision
==================================

In delta form, each SDC sweep solves for the correction of the iterate instead of the iterate itself.
An implicit solver that returns the correction with an error relative to its size introduces an error
that vanishes as the iteration converges. This project uses that to run implicit solves, coarse levels
and stored states at reduced precision, and to predict which precision a node-local solve may use.

Usage
-----

``delta_implicit`` replaces ``generic_implicit`` and gives the same iterates:

.. code-block:: python

    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    description['sweeper_class'] = delta_implicit

The node-local solve works on the correction in one of two ways:

- **Linear or affine implicit part:** set ``linear_implicit=True`` in ``sweeper_params``. The problem's
  ``solve_system`` then receives the right-hand side of the correction equation.
- **Nonlinear implicit part:** the problem class provides ``solve_system_delta(r, factor, base, f_base,
  t)``, which solves :math:`\delta - \text{factor}\,[f(\text{base} + \delta) - f(\text{base})] = r` and
  returns :math:`\delta`. The increment has to be evaluated without cancellation, see
  ``allencahn_delta`` in ``problems.py``.

Without either, the sweeper solves for :math:`u^k_m + \delta_m` as ``generic_implicit`` does, and the
precision of the solve is not decoupled from the precision of the result. For MLSDC and PFASST, set
``base_transfer_class`` to ``delta_transfer`` from
``pySDC.implementations.transfer_classes.BaseTransferDelta``.

Contents
--------

The sweepers and transfer classes are part of pySDC (``pySDC/implementations``):

============================================  =======================================
``sweeper_classes/delta_form.py``             the sweepers, at any number of levels
``sweeper_classes/delta_form_MPI.py``         the node-parallel sweepers
``transfer_classes/BaseTransferDelta.py``     the transfer between levels
``transfer_classes/BaseTransferDeltaMPI.py``  the node-parallel transfer
============================================  =======================================

The optional problem methods they use, ``eval_f_increment`` and ``solve_system_delta``, are documented
on :class:`pySDC.core.problem.Problem`. This project contains:

=======================  ===========================================================================
``problems.py``          ``allencahn_delta`` (nonlinear), ``heat_delta`` (linear) and
                         ``vanderpol_delta``
``problems_petsc.py``    ``petsc_fisher_delta``, reduced precision emulated
``problems_fenics.py``   ``fenics_grayscott_delta``, reduced precision emulated
``inexact.py``           solves with a given relative accuracy, or with results rounded to a format
``genuine.py``           solves computed in fp32, fp16 or bf16 arithmetic
``precision_theory.py``  predicted additional iterations and admissible formats, per iteration
``eta_spec.py``          the test cases, and the measured accuracy a solve needs
``work_precision.py``    work and error for a target accuracy, with formats chosen per step
``mlsdc.py``             reduced precision on a level, emulated
``cascade.py``           storage precision raised as the iteration converges
``sweepers_MPI.py``      the same, for the node-parallel sweepers
``paradiag.py``          ParaDiag's transform at reduced precision
``run_demo.py``          demonstration runs, nonlinear and linear
``run_gpu.py``           runs on CPU or GPU, including the half-precision FFT
``run_mpi.py``           node-parallel runs, on three ranks
``run_petsc.py``         runs with PETSc
``run_fenics.py``        runs with FEniCS
=======================  ===========================================================================

Tests
-----

.. code-block:: bash

    pytest pySDC/projects/DeltaSDC/tests

The tests that need FEniCS, PETSc, mpi4py or a GPU are in ``pySDC/tests/test_projects/test_DeltaSDC``,
because the CI jobs that install these backends select tests by marker. They call the ``run_*.py``
scripts.
