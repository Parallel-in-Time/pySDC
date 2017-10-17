Step-6: Advanced PFASST controllers
===================================

We discuss controller implementations and features besides the standard classic and serial PFASST controller in this step.

Part A: Classical vs. multigrid controller
------------------------------------------

Besides the ``allinclusive_classic_nonMPI`` controller we have used so far, pySDC comes with (at least) three more controllers.
While we do not discuss MPI-based controllers here, the other major branch is the multigrid controller.
In contrast to the classical scheme, this implementation of PFASST does not overlap communication and computation as the classical implementation does.
It resembles more closely a multigrid-in-time algorithm by performing each stage for all processes at once (i.e. fine sweep, restrict, coarse sweep, interpolate, etc.).
Consequently, processes can only finish all at once.

Important things to note:

- If only SDC or MLSDC are run, classical and multigrid controller do not differ.
  The difference is only in the communication scheme and the stopping criterion for multiple processes.
- One major advantage of having the multigrid controller at hand is that some MPI implementations on certain machines do not work well with overlapping.
- The multigrid controller cannot handle multi-step SDC (see step 7) due to design decisions.

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
To do this, pySDC comes with the two MPI-parallelized controllers, namely ``allinclusive_classic_MPI`` and ``allinclusive_multigrid_MPI``.
Both are supposed to yield the same results as their non-MPI counterparts and this is what we are demonstrating here (at least for one particular example).
The actual code of this part is rather short, since the only task is to call another snippet (``playground_parallelization.py``) with different number of parallel processes.
This is realized using Python's ``subprocess`` library and we check at the end if each call returned normally.
Now, the snippet called by the example is the basically the same code as use by Parts A and B.
We can use the results of Parts A and B to compare with and we expect the same number of iterations, the same accuracy and the same difference between the two flavors as in Part A (up to machine precision).

Important things to note:

- The additional Python script ``playground_parallelization.py`` contains the code to run the MPI-parallel controllers. To this end, we import the routine ``set_parameters`` from Part A to ensure that we use teh same set of parameters for all runs.
- This example also shows how the statistics of multiple MPI processes can be gathered and processed by rank 0, see ``playground_parallelization.py``.
- Both controllers need a working installation of ``mpi4py``. Since this is not always easy to achieve and since debugging a parallel program can cause a lot of headaches, the non-MPI controllers perform the same operations in serial.
- The somewhat weird notation with the current working directory ``cwd`` is due to the corresponding test, which, run by nosetests, has a different working directory than the tutorial.

.. include:: doc_step_6_C.rst
