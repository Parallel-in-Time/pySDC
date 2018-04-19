Second-order Problems
=====================

In this project, we run several examples of the `second-order Verlet-SDC integrator <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/sweeper_classes/verlet.py>`_.
This SDC variant is the core integrator for the `Boris method <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/sweeper_classes/boris_2nd_order.py>`_, which we use e.g. in the 3rd tutorial.

Simple problems
---------------

We first test the integrator for some rather simple problems, namely the harmonic oscillator and the Henon-Heiles problem.
For both problems we make use of the hook ``hamiltonian_output`` to monitor the deviation from the exact Hamiltonian.
PFASST is run with 100 processors (virtually parallel) and the deviation from the initial Hamiltonian is shown.

.. include:: doc_hamiltonian_simple.rst

Solar system problem
--------------------

In this slightly more complex setup we simulate the movement of planets in the solar system.
The acceleration due to gravitational forces are computed using a simple N^2 solver.
We can use two different setups:

-  ``OuterSolarSystem`` problem class: only the six outer planets are simulated, namely the Sun (which in its mass contains the inner planets), Jupiter, Saturn, Uranus, Neptune and Pluto.
-  ``FullSolarSystem`` problem class: all planets are simulated, with earth and moon combined

Coarsening can be done using only the sun for computing the acceleration.
Note how PFASST works very well for the outer solar system problem, but not so well for the full solar system problem.
Here, over 15 iterations are required in the mean, while SDC and MLSDC require only about 5 per step.

.. include:: doc_solar_system.rst