Step-6: Advanced topics
=======================

Since we now have explored the basic features of pySDC, we are actually ready to do some serious science (e.g. in the projects).
However, we gather here further interesting cases, e.g. special flags or more alternative implementations of components.

Part A: Visualizing Residuals
-----------------------------

In this part, we briefly introduce the visualization of residuals, built in into pySDC's plugins.
The application is (supposed) to be simple: merely put the ``stats`` object into the function ``show_residual_across_simulation`` and look at the resulting figure.

Important things to note:

- The function visualizes simply the residuals over all processes and all iterations, but only for a single block.
- The function itself is pretty straightforward and does not require passing the number of processes or iterations.

Full code: `tutorial/step_6/A_visualize_residuals.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_6/A_visualize_residuals.py>`_

.. literalinclude:: ../../../tutorial/step_6/A_visualize_residuals.py

Results:

.. literalinclude:: ../../../step_6_A_out.txt

.. image:: ../../../step_6_residuals.png
   :scale: 50 %

Part B: Classical vs. multigrid controller
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
- The multigrid controller cannot handle multi-step SDC (see Part C) due to design decisions.

Full code: `tutorial/step_6/B_classic_vs_multigrid_controller.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_6/B_classic_vs_multigrid_controller.py>`_

.. literalinclude:: ../../../tutorial/step_6/B_classic_vs_multigrid_controller.py

Results:

.. literalinclude:: ../../../step_6_B_out.txt

.. image:: ../../../step_6_residuals_multigrid.png
   :scale: 50 %

Part C: Multi-step SDC
----------------------


Part X: To be continued...
--------------------------

We shall see what comes next...