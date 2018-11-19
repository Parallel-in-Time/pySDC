Step-7: Advanced topics
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

.. include:: doc_step_7_A.rst

Part B: Multi-step SDC
----------------------

One interesting question when playing around with the different configurations is this: what happens, if we want parallel time-steps but only a single level?
The result is called multi-step SDC. Here, after each sweep the result is sent forward, but is picked up in the next (and not current) iteration.
This corresponds to performing only the smoother stage in a multigrid scheme.
Parallelization is dead-simple and no coarsening strategy is needed.
Yet, the missing stabilization of the coarse level leads to a significant increase in iterations, when more time-steps are computed in parallel.
To prevent this, information can be sent forward immediately, but then this is not a parallel algorithm anymore..

Important things to note:

- Use the controller parameter ``mssdc_jac`` to controll whether the method should be "parallel" (Jacobi-like) or "serial" (Gauss-like).
- We increased the logging value here again, (safely) ignoring the warnings for multi-step SDC.

.. include:: doc_step_7_B.rst

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

Part X: To be continued...
--------------------------

We shall see what comes next...