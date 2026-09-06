Step-9: ParaDiag
================

ParaDiag is a parallel-in-time method of a rather different flavour than PFASST.
Instead of iterating on a hierarchy of levels and passing information forward step by step, it diagonalizes the "top layer" of Kronecker products that makes up the composite collocation problem.
After the diagonalization, the collocation problems on the individual steps decouple and can be solved concurrently, which is where the parallelism comes from.

The price is an approximation: the time-stepping matrix is replaced by an :math:`\alpha`-circulant one, which *is* diagonalizable by a weighted Fourier transform.
The outer iteration then corrects for that perturbation, and :math:`\alpha` trades approximation quality against the conditioning of the diagonalization.

Part A: ParaDiag for linear problems
------------------------------------

We start with the linear case, where the composite collocation problem really can be written as a matrix and the whole method is a few lines of linear algebra.
It is recommended to view this code side by side with `Gaya's paper on ParaDiag with collocation methods <https://arxiv.org/abs/2103.12571>`_, as the code follows the equations there closely without repeating their explanation.

Important things to note:

- The diagonalization happens across the time-steps, not across the collocation nodes.
- The :math:`\alpha`-circulant approximation is what makes the diagonalization possible in the first place.

.. include:: doc_step_9_A.rst

Part B: ParaDiag for nonlinear problems
---------------------------------------

For nonlinear problems the composite collocation problem cannot be written as a matrix, so the diagonalization needs a linear operator to work with.
This part shows the two ways out: IMEX splitting, where only the linear implicit part enters the preconditioner, and averaging the Jacobian across the steps.

Important things to note:

- Averaging the Jacobian requires communicating the average solution, which is why ``average_jacobian`` is off by default for linear problems.
- We do a single Newton iteration per ParaDiag iteration, so the number of Newton iterations per node equals the number of ParaDiag iterations.

.. include:: doc_step_9_B.rst

Part C: ParaDiag in pySDC
-------------------------

Here we leave the hand-written linear algebra behind and set ParaDiag up through pySDC's controllers, comparing it to single-level PFASST in Jacobi mode and to serial time stepping.
Both schemes are used without any optimization, so please refrain from computing parallel efficiency from these numbers.

Important things to note:

- ParaDiag needs its own sweeper (``QDiagonalization``) and its own controller.
- The solution becomes complex, because the diagonalization is.
- ParaDiag converges in very few iterations for the hyperbolic advection example, where PFASST struggles, and the picture reverses for the van der Pol oscillator.

.. include:: doc_step_9_C.rst

Part D: MPI-parallel ParaDiag
-----------------------------

Parts A to C all ran ParaDiag with the "virtually parallel" controller, which keeps every step in a single process.
That is what you want while developing, but it does not actually run in parallel.
This part uses ``controller_ParaDiag_MPI`` instead, with one time-step per rank and the communicator spanning the block that is diagonalized.
Nothing about the method changes: the description and the controller parameters are the same, only the controller class differs.

We always integrate the same total number of time-steps and only vary how many of them run in parallel.
With four steps in total and a block size of one, two or four, the controller has to window through four, two or one block respectively.
Windowing works the same way in both controllers, so we run each block size with the MPI controller *and* the virtually parallel one and compare.

Important things to note:

- All steps of a block iterate together. In PFASST an early step can converge and drop out, which is what makes it pipelined; ParaDiag cannot do that, because the transform in time needs every step. A step that stopped early would leave the others waiting.
- Consequently the block is always full. If the end time does not divide into whole blocks, ParaDiag solves past it rather than truncating, and says so.
- Neither the iteration counts nor the error depend on how the time domain is split into blocks, which is what we check here. Spreading a block over more processes must not change the method, and windowing through more blocks must not either.
- ``alpha`` may also be a list or a callable of the iteration index, if you want to start with a well-conditioned value and tighten it later.

.. include:: doc_step_9_D.rst

Part E: Adaptive alpha
----------------------

Every part so far picked :math:`\alpha` by hand and kept it fixed, which means committing to one compromise for the whole run.
A small :math:`\alpha` approximates the original problem better and converges in fewer iterations, but conditions the diagonalization worse, so round-off and inexact inner solves get amplified.
The right balance shifts as the residual falls, so a fixed value is wrong at one end of the run or the other.

The ``AdaptiveAlpha`` convergence controller updates :math:`\alpha` after every iteration instead, following `Čaklović et al. <https://doi.org/10.2140/camcos.2023.18.55>`_:

.. math::
    \gamma = L (3 \epsilon + \tau), \quad
    \alpha_{k} = \sqrt{\frac{\gamma r_k}{e_k}}, \quad
    e_{k+1} = 2 \sqrt{\gamma e_k r_k},

with :math:`L` the number of steps in the block, :math:`\epsilon` machine precision, :math:`\tau` the inner solver tolerance, :math:`r_k` the residual and :math:`e_k` a running bound on the error.

Important things to note:

- :math:`\gamma` is an accuracy floor. There is no point pushing :math:`\alpha` below the level at which round-off and the inner solver dominate anyway, which is why ``inner_tol`` enters: a looser inner solve should get a larger :math:`\alpha`.
- The interesting result is not that the adaptive strategy wins on iteration count. It ties with the best fixed value we tried, but it gets there without being told, and it keeps :math:`\alpha` orders of magnitude larger while doing so, which is exactly the margin that protects you once the inner solves are inexact.
- The residual is taken over the whole block, so every rank computes the same :math:`\alpha` and the controllers stay in step.
- :math:`\alpha` is a property of the method, not of the parallelization. The adaptive controller therefore has to produce the same iteration counts whether ParaDiag runs virtually or across MPI ranks, which is what we check here.

.. include:: doc_step_9_E.rst
