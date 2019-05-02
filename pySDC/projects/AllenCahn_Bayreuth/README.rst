Allen-Cahn problems from Bayreuth
=================================

This project provides code for running, testing, benchmarking and playing with Allen-Cahn-type problems in material science.
The setups come from `ParaPhase <http://paraphase.de>`_ partner `Bayreuth <https://www.metalle.uni-bayreuth.de>`_ and the codes should demonstrate correctness, parallelization and flexibility of the implementation with pySDC.

Verification
------------

The script ``run_simple_forcing_problems.py`` tests various 2D test problems and compares the results against known values.
This is done by setting up a circle of value 1, which then shrinks. The speed of this process is known for different setups.
In detail, the code checks:

- no driving force: circle vanishes after a certain number of steps, numerical scheme should not be too "slow"
- constant driving force: for a particular value of the driving force ``dw``, the circle's radius stays constant (more or less)
- time-dependent driving force: weighting the RHS and the driving force allows to determine a force to keep the radius constant, too

Further scripts/files
---------------------

- ``AllenCahn_monitor_and_dump.py``: computes the radii for the verification problems and dumps the solution via MPI I/O
