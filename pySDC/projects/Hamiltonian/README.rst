Attempts to parallelize SDC
===========================

In this project, we run several examples of the second-order Verlet-SDC integrator.
This SDC variant is the core integrator for the Boris method, which we use e.g. in the 3rd tutorial.

Simple problems
---------------

We first test the integrator for some rather simple problems, namely the harmonic oscillator and the Henon-Heiles problem.
For both problems we make use of the hook ``hamiltonian_output`` to monitor the deviation from the exact Hamiltonian.

.. include:: doc_hamiltonian_simple.rst
