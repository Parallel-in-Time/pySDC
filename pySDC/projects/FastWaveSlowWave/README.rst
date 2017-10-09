Fast-Wave-Slow-Wave SDC
=======================

In this project, we explore semi-implicit spectral deferred corrections (SISDC) in which the stiff, fast dynamics correspond to fast propagating waves.
We study the performance of the method compared to standard integrators like RK-IMEX or DIRK schemes and analyze the convergence properties for scalar test problems.
This project contains the code for the publication `Spectral deferred corrections with fast-wave-slow-wave splitting <http://dx.doi.org/10.1137/16M1060078>`_ of pySDC v2,
while the original code can be found under `The fast-wave-slow-wave release, v2 <https://doi.org/10.5281/zenodo.53849>`_.
Note that due to the long runtime, not all results are generated via Travis.
For the Boussinesq example and the convergence test of the acoustic-advection equation, only the visualization (and therefore the existence of the data files) is tested.
We omit the codes in this documentation, since they are rather long and slightly complex.

Theoretical results
-------------------

Here, we review FWSW-SDC from two different viewpoints: as a split method with a fixed order set by a fixed number of iterations K for a sufficiently large number of nodes M,
or as an iterative solver for the collocation problem where iterations are performed until the norm of the residual reaches a prescribed tolerance.
We investigate fwsw-SDC from both viewpoints for the scalar test problem and analyze:

- the `spectral radius <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/plot_stifflimit_specrad.py>`_ of the iteration matrix (Fig. 1 in the above mentioned publication)
- the `stability domains <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/plot_stability.py>`_ of different configurations (Fig. 2)
- `stability <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/plot_stab_vs_k.py>`_ with respect to the iteration number k (Fig. 3)
- the `dispersion relation <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/plot_dispersion.py>`_ (Fig. 4)

.. include:: doc_fwsw_theory.rst

Acoustic-advection example
--------------------------

In a first more complex example, we consider the 1D acoustic-advection example. We show:

- `convergence <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/runconvergence_acoustic.py>`_ of FWSW-SDC with orders 3, 4, and 5 versus number of time steps (Fig. 5, left)
- `convergence rate <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/runitererror_acoustic.py>`_ of the FWSW-SDC iteration (Fig. 5, right)
- the numerical solution of the acoustic-advection equation with `multiscale initial data <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/runmultiscale_acoustic.py>`_ (Fig. 6)

.. include:: doc_fwsw_acoustic.rst

Boussinesq example
------------------

In a second, even more complex example, we test FWSW-SDC for the 2D Boussinesq equation.
In particular, we are interested in the number of `GMRES iterations <https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/FastWaveSlowWave/rungmrescounter_boussinesq.py>`_ each time integrator needs to achieve a certain error.

.. include:: doc_fwsw_boussinesq.rst

