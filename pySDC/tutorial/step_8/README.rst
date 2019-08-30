Step-8: Advanced topics
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

.. include:: doc_step_8_A.rst

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

.. include:: doc_step_8_B.rst

Part C: Iteration estimator
---------------------------

One may ask when to stop the SDC, MLSDC, PFASST iterations. So far, we have used a residual threshold or a fixed number of iterations to stop the process.
Another option is to estimate the number of iterations it takes to reach a given error w.r.t. the exact collocation solution.
This can be done by using two consecutive iterates to estimate the Lipschitz constant of the iteration procedure.
Adding a few magic/safety constants here and there and you can guess when to stop.
This example takes three different test cases and checks how well the iteration estimator drives the iterates below the threshold.

Important things to note:

- The estimator also works for PFASST, where is ensures that up to each step (!) the tolerance is met.
- The method also works for the parallel `controller_MPI` controller by using interrupts for checking when to stop (not tested here).

.. include:: doc_step_8_C.rst

Part X: To be continued...
--------------------------

We shall see what comes next...