Step-6: Multigrid and MPI parallelization
=========================================


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

Full code: `tutorial/step_6/A_classic_vs_multigrid_controller.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_6/A_classic_vs_multigrid_controller.py>`_

.. literalinclude:: ../../../tutorial/step_6/A_classic_vs_multigrid_controller.py

Results:

.. literalinclude:: ../../../step_6_A_out.txt


Part B: MPI parallelization
---------------------------

