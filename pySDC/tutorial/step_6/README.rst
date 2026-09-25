Step-6: Advanced PFASST controllers
===================================

We discuss controller implementations, features and parallelization of PFASST controllers in this step.

Part A: The nonMPI controller
------------------------------------------

pySDC comes with (at least) two controllers: the standard, non-MPI controller we have used so far and the MPI_parallel one.
The nonMPI controller can be used to run simulations without having to worry about parallelization and MPI installations.
By monitoring the convergence, this controller can already give a detailed idea of how PFASST will work for a given problem.

Important things to note:

- If you don't want to deal with parallelization and/or are only interested in SDC, MLSDC or convergence of PFASST, use the nonMPI controller.
- If you care for parallelization, use the MPI controller, see Part C.

.. include:: doc_step_6_A.rst

Part B: Odd temporal distribution
---------------------------------

Accidentally, the numbers of parallel processes used in Part A are always dividers of the number of steps.
Yet, this does not need to be the case. All controllers are capable of handling odd distributions, e.g. too few or too many processes for the steps (or for the las block).
This is demonstrated here, where the code from Part A is called again with odd number of parallel steps.

Important things to note:

- This capability may become useful if adaptive time-stepping is used. The controllers check for currently active steps and only those will compute the next block.
- This also works for/with SDC and MLSDC, where in the case of varying time-step sizes the overall number of steps is not given at the beginning.

.. include:: doc_step_6_B.rst

Part C: MPI parallelization
---------------------------

Since PFASST is actually a parallel algorithm, executing it in parallel e.g. using MPI might be an interesting exercise.
To do this, pySDC comes with the MPI-parallelized controller, namely ``controller_MPI``.
It is supposed to yield the same results as the non-MPI counterpart and this is what we are demonstrating here (at least for one particular example).
The code is the same as in Parts A and B -- it imports ``set_parameters`` from Part A so that all runs use the same parameters -- with ``controller_MPI`` in place of the non-MPI controller.

Run it as you would run any MPI program, with one rank per parallel step::

    mpirun -np 4 python C_MPI_parallelization.py

The number of parallel steps is simply the size of ``MPI.COMM_WORLD``, so there is nothing to configure: run it on 4 ranks for 4 parallel steps, on 3 for 3, and so on.
We can use the results of Parts A and B to compare with and we expect the same number of iterations, the same accuracy and the same difference between the two flavors as in Part A (up to machine precision).

Important things to note:

- This example also shows how the statistics of multiple MPI processes can be gathered and processed by rank 0.
- The controller needs a working installation of ``mpi4py``. Since this is not always easy to achieve and since debugging a parallel program can cause a lot of headaches, the non-MPI controller performs the same operations in serial.
- The test that covers this part runs the same file on 1, 2, 3, 4, 5, 7, 8 and 9 ranks through `mpi-pytest <https://github.com/firedrakeproject/mpi-pytest>`_, which is also how the rest of pySDC's MPI tests are run.

.. include:: doc_step_6_C.rst
