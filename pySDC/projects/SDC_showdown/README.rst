What is the fastest SDC variant?
================================

In this project, we test different variants of SDC for a particular problem to see which one is the fastest.
More precisely, we run SDC for the 1D Fisher equation and the 2D Gray-Scott problem with PETSc data types and solvers, using fully implicit, semi-implicit and multi-implicit time-stepping.
We also test exact spatial solves vs. inexact ones (aka inexact SDC, ISDC)

Fisher and Gray-Scott equations
-------------------------------

The two run scripts simply test all variants of SDC after the other, each of them exact and then inexact.
The results are gathered, stored and shown in comparison.
Note that on standard machines the inexact semi-implicit variant wins each time (may be different for the CI testing).

.. include:: doc_SDC_showdown.rst

