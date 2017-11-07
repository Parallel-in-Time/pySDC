Matrix-based versions of PFASST
===============================

In this project, we use and test two matrix-based version of PFASST for linear problems:
we write PFASST as a standard twogrid method and we derive the propagaton matrix of PFASST.
Both approaches are compared to the standard multigrid controller of PFASST and with each other.
This includes tests with the heat, the advection as well as the test equation.

Matrix-based PFASST
-------------------

When written as a multigrid method for a linear composite collocation problem, PFASST can be compactly defined via a
smoothing and a coarse-grid correction step, just as a standard two-grid method. In the new (and very specialized)
controller ``allinclusive_matrix_nonMPI.py``, this concept is exploited to obtain a PFASST (and MLSDC and SDC)
controller closely resembling this idea and notation. In ``compare_to_matrixbased.py``, this controller is tested
against the standard PFASST implementation.

.. include:: doc_matrixPFASST_matrix.rst

Propagator-based PFASST
-----------------------

The second approach follows directly from the matrix formulation: instead of writing PFASST as an iterative scheme,
we now derive the full propagation matrix, which takes the initial value u0 and produces the final value uend,
over all steps, iterations and sweeps. This matrix can be used to analyze PFASST in yet another exciting way.

.. include:: doc_matrixPFASST_propagator.rst

