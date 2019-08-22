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

Part X: To be continued...
--------------------------

We shall see what comes next...