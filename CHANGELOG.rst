Changelog
---------

- August 30, 2019: Version 3.1 adds many more examples like the nonlinear Schrödinger equation, more on Gray-Scott and in particular Allen-Cahn.
  Those are many implemented using the parallel FFT library `mpi4pi-fft <https://bitbucket.org/mpi4py/mpi4py-fft/src/master/>`_, which can now be used with pySDC.
  There are now 8 tutorials, where step 7 shows the usage of three external libraries with pySDC: mpi4py, FEniCS and petsc4py.
  The MPI controller has been improved after performaning a detailed performance analysis using `Score-P <https://www.vi-hps.org/projects/score-p/>`_ and `Extrae <https://www.vi-hps.org/Tools/Extrae.html>`_.
  Finally: first steps towards error/iteration estimators are taken, too.

- February 14, 2019: Released version 3 of `pySDC`. This release is accompanied by the **ACM TOMS paper**
  `"pySDC --  Prototyping spectral deferred corrections" <https://doi.org/10.1145/3310410>`_.
  It release contains some breaking changes to the API. In detail:

  - **Dropped Python 2 support**: Starting with this version, `pySDC` relies on Python 3. Various incompabilities led
    to inconsistent treatment of dependencies, so that parts of the code had to use Python 2 while other relied on
    Python 3 or could do both. We follow `A pledge to migrate to Python 3 <https://python3statement.org/>`_ with this decision,
    as most prominent dependencies of `pySDC` already do.
  - **Unified controllers**: Instead of providing (and maintaining) four different controllers, this release only has
    one for emulated and one for MPI-based time-parallelization (``controller_nonMPI`` and ``controller_MPI``).
    This should avoid further confusion and makes the code easier to maintain. Both controllers use the multigrid
    perspective for the algorithm (first exchange data, than compute updates), but the classical way of determining
    when to stop locally (each time-step is stopped when ready, if the previous one is ready, too). The complete multigrid
    behavior can be restored using a flag. All included projects and tutorials have been adapted to this.
  - **No more data types in the front-ends**: The redundant use of data type specifications in the description dictionaries
    has been removed. Data types are now declared within each problem class (more precisely, in the header of the
    ``__init__``-method to allow inhertiance). All included projects and tutorials have been adapted to this.
  - **Renewed FEniCS support**: This release revives the deprecated `FEniCS <https://fenicsproject.org/>`_ support, now requiring at least FEniCS 2018.1.
    The integration is tested using Travis-CI.
  - **More consistent handling of local initial conditions**: The treatment of ``u[0]`` and ``f[0]`` has been fixed and
    made consistent throughout the code.
  - As usual, many bugs have been discovered and fixed.

- May 23, 3018: Version 2.4 adds support for `petsc4py <https://bitbucket.org/petsc/petsc4py>`_!
  You can now use `PETSc <http://www.mcs.anl.gov/petsc/>`_ data types (`pySDC` ships with DMDA for distributed structured grids) and parallel solvers right from your examples and problem classes.
  There is also a new tutorial (7.C) showing this in a bit more detail, including communicator splitting for parallelization in space and time.
  Warning: in order to get this to work you need to install petsc4py and mpi4py first! Make sure both use MPICH3 bindings.
  Downloading `pySDC` from PyPI does not include these packages.

- February 8, 2018: Ever got annoyed at `pySDC`'s incredibly slow setup phase when multiple time-steps are used? Version 2.3
  changes this by copying the data structure of the first step to all other steps using the `dill Package <https://pypi.python.org/pypi/dill>`_.
  Setup times could be reduced by 90% and more for certain problems. We also increase the speed for certain calculations,
  in particular for the Penning trap example.

- November 7, 2017: Version 2.2 contains matrix-based versions of PFASST within the project ``matrixPFASST``. This involved quite a few
  changes in more or less unexpected places, e.g. in the multigrid controller and the transfer base class. The impact
  of these changes on other projects should be negligible, though.

- October 25, 2017: For the `6th Workshop on Parallel-in-Time Integration <https://www.ics.usi.ch/index.php/6th-workshop-on-parallel-in-time-methods>`_
  `pySDC` has been updated to version 2.1. It is now available on PyPI - the Python Package Index, see `https://pypi.python.org/pypi/pySDC <https://pypi.python.org/pypi/pySDC>`_
  and can be installed simply by using ``pip install pySDC``. Naturally, this release contains a lot of bugfixes and minor improvements.
  Most notably, the file structure has been changed again to meet the standards for Python packaging (at least a bit).

- November 24, 2016: Released version 2 of `pySDC`. This release contains major changes to the code and its structure:

  - **Complete redesign of code structure**: The ``core`` part of `pySDC` only contains the core modules and classes,
    while ``implementations`` contains the actual implementations necessary to run something.
    This now includes separate files for all collocation classes, as well as a collection of problems, transfer classes and so on.
    Most examples have been ported to either ``tutorials``, ``playgrounds`` or ``projects``.

  - **Introduction of tutorials**: We added a tutorial (see below) to explain many
    of pySDC's features in a step-by-step fashion. We start with a simple spatial
    discretization and collocation formulations and move step by step to SDC, MLSDC and PFASST.
    All tutorials are accompanied by tests.

  - **New all-inclusive controllers**: Instead of having two PFASST controllers
    which could also do SDC and MLSDC (and more), we now have four generic controllers
    which can do all these methods, depending on the input. They are split into
    two by two class: `MPI` and `NonMPI` for real or virtual parallelisim as well
    as `classic` and `multigrid` for the standard and multigrid-like implementation
    of PFASST and the likes. Initialization has been simplified a lot, too.

  - **Collocation-based coarsening** As the standard PFASST libraries `libpfasst <https://bitbucket.org/memmett/libpfasst>`_ and `PFASST++ <https://github.com/Parallel-in-Time/PFASST>`_
    `pySDC` now offers collocation-based coarsening, i.e. the number of collocation nodes can be reduced during coarsening.
    Also, time-step coarsening is in preparation, but not implemented yet.

  - **Testing and documentation** The core, implementations and plugin packages and their subpackages are fully documented using sphinx-apidoc, see below.
    This documentation as well as this website are generated automatically using `Travis-CI <https://travis-ci.org/Parallel-in-Time/pySDC>`_.
    Most of the code is supported by tests, mainly realized by using the tutorial as the test routines with clearly defined results. Also, projects are accompanied by tests.

  - Further, minor changes:

    - Switched to more stable barycentric interpolation for the quadrature weights
    - New collocation class: `EquidistantSpline_Right` for spline-based quadrature
    - Collocation tests are realized by generators and not by classes
    - Multi-step SDC (aka single-level PFASST) now works as expected
    - Reworked many of the internal structures for consistency and simplicity