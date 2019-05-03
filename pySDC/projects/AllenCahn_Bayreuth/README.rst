Allen-Cahn problems from Bayreuth
=================================

This project provides code for running, testing, benchmarking and playing with Allen-Cahn-type problems in material science.
The setups come from `ParaPhase <http://paraphase.de>`_ partner `Bayreuth <https://www.metalle.uni-bayreuth.de>`_ and the codes should demonstrate correctness, parallelization and flexibility of the implementation with pySDC.
Many (possibly all) codes in this project need `mpi4py-fft <https://mpi4py-fft.readthedocs.io/en/latest/>`_ which can be installed via pip or conda.

Verification
------------

The script ``run_simple_forcing_verification.py`` tests various 2D test problems and compares the results against known values.
This is done by setting up a circle of value 1, which then shrinks. The speed of this process is known for different setups.
In detail, the code checks:

- no driving force: circle vanishes after a certain number of steps, numerical scheme should not be too "slow"
- constant driving force: for a particular value of the driving force ``dw``, the circle's radius stays constant (more or less)
- time-dependent driving force: weighting the RHS and the driving force allows to determine a force to keep the radius constant, too

The script ``run_temp_forcing_verification.py`` runs a specific setup with temperature-based forcing. The datatype now has two components.
Verification is done using a numerical simulation created with ``run_temp_forcing_reference.py``. Errors and order of accuracy is tested.

These small-scale problem setups are all tested with CI.
They run serial or parallel in space (although testing for the temperature setup only works in serial), but serial in time (mimicking PFASST, though).

Benchmark
---------

The script ``run_simple_forcing_verification.py`` can be used to benchmark the code for simple driving forces on larger HPC machines.
It takes the number of processes in space as well as the setup type (as in the verification code) as input parameters.
It can be run serial/parallel in space and/or time.

These larger-scale codes are not tested with CI.

Application
-----------

TODO

Further/related files
---------------------

The following files are also listed under this project:

- ``AllenCahn_monitor_and_dump.py``: computes the radii for the verification problems and dumps the solution via MPI I/O
- ``AllenCahn_monitor.py``: computes the radii for the verification problems
- ``AllenCahn_dump.py``: dumps the solution via MPI I/O
- ``visualize.py``: simple script to turn field data from the dump routines into pngs
- ``visualize_temp.py``: simple script to turn field and temperature data from the dump routines into pngs and movies
- ``run_pySC_AC.exe``: jobscript for JURECA

The project uses in particular these implementations from the ``implementations`` folder:

- ``parallel_mesh.py`` in the ``datatype_classes``: contains numpy-based data with a communicator
- ``AllenCahn_MPIFFT.py`` in the ``problem_classes``: contains the implementation of the problem for simple forcing terms
- ``AllenCahn_Temp_MPIFFT.py`` in the ``problem_classes``: contains the implementation of the problem for temperature-based forcing

