Step-6: Advanced PFASST controllers
===================================

We discuss controller implementations, features and parallelization of PFASST controller in this step.

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
The actual code of this part is rather short, since the only task is to call another snippet (``playground_parallelization.py``) with different number of parallel processes.
This is realized using Python's ``subprocess`` library and we check at the end if each call returned normally.
Now, the snippet called by the example is the basically the same code as use by Parts A and B.
We can use the results of Parts A and B to compare with and we expect the same number of iterations, the same accuracy and the same difference between the two flavors as in Part A (up to machine precision).

Important things to note:

- The additional Python script ``playground_parallelization.py`` contains the code to run the MPI-parallel controller. To this end, we import the routine ``set_parameters`` from Part A to ensure that we use the same set of parameters for all runs.
- This example also shows how the statistics of multiple MPI processes can be gathered and processed by rank 0, see ``playground_parallelization.py``.
- The controller need a working installation of ``mpi4py``. Since this is not always easy to achieve and since debugging a parallel program can cause a lot of headaches, the non-MPI controller performs the same operations in serial.
- The somewhat weird notation with the current working directory ``cwd`` is due to the corresponding test, which, run by nosetests, has a different working directory than the tutorial.

.. include:: doc_step_6_C.rst
