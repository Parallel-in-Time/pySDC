# Changelog

:arrow_left: [Back to main page](./README.md)

-   October 7, 2022: Version 5 comes with many changes, both visible and
    invisible ones. Some of those break the existing API, but if you are
    using tests, you should be fine. Major changes include:

    -   **New convergence controllers**: Checking whether a step has
        converged can be tricky, so we made separate modules out of
        these checks. This makes features like adaptivity easier to
        implement. Also, the controllers have been streamlined a bit to
        make them more readable/digestible. Thanks to
        [\@brownbaerchen](https://github.com/brownbaerchen)!
    -   **Adaptivity and error estimators**: SDC now comes with
        adaptivity and error estimation, leveraging the new convergence
        controllers out of the box. Thanks to
        [\@brownbaerchen](https://github.com/brownbaerchen)!
    -   **New collocation classes**: We completely rewrote the way
        collocation nodes and weights are computed. It is now faster,
        more reliable, shorter, better. But: this **breaks the API**,
        since the old collocation classes do not exist anymore. The
        projects, tutorials, tests and most of the playgrounds have been
        adapted, so have a look over there to see [what to
        change](https://github.com/Parallel-in-Time/pySDC/commit/01ffabf71a8d71d33b74809271e8ad5a7b03ac5e#diff-adf74297b6c64d320f4da0f1d5528eda6229803a6615baf5d54c418032543681).
        Thanks to [\@tlunet](https://github.com/tlunet)!
    -   **New projects**: Resilience and energy grid simulations are
        ready to play with and are waiting for more ideas! We used this
        effort to condense and clean up the problem classes a bit,
        reducing the number of files and classes with only marginal
        differences significantly. This could potentially **break your
        code**, too, if you rely on any of those affected ones. Thanks
        to [\@brownbaerchen](https://github.com/brownbaerchen) and
        [\@lisawim](https://github.com/lisawim)!
    -   **Toward GPU computing**: We included a new data type based on
        [CuPy](https://cupy.dev/) making GPU computing possible. Thanks
        to [\@timo2705](https://github.com/timo2705)!
    -   **Better testing**: The CI pipeline got a complete overhaul
        (again), now enabling simultaneous tests, faster/earlier
        linting, benchmarking (at least, in principal), separate
        environments and so on. The code is tested under Ubuntu and
        MacOS.
    -   **Better code formatting**: `pySDC` now uses
        [black](https://black.readthedocs.io) and
        [flakeheaven](https://flakeheaven.readthedocs.io) for cleaner
        source code. After complaints here and there about linting
        \"errors\" the recommended way now is to run `black` before
        submission.

-   December 13, 2021: Version 4.2 brings compatibility with Python 3.9,
    including some code cleanup. The CI test suite seems to run faster
    now, since sorting out the dependencies is faster. Tested
    [mamba](https://github.com/mamba-org/mamba), which for now makes the
    CI pipeline much faster. Also, the CI workflow can now run locally
    using [act](https://github.com/nektos/act). We introduced markers
    for soem of the tests in preparation of distributed tests on
    different platforms. And finally, a LaTeX installation is no longer
    needed use plotting (but recommended).

-   August 11, 2021: Version 4.1 has some more changes under the hood,
    most of them with no significant impact to users. The CI pipeline
    has been completely rewritten, porting the code to [Github
    Actions](https://github.com/features/actions) (away from [Travis
    CI](https://travis-ci.com/)), to [flake8](https://flake8.pycqa.org)
    and to [pytest](https://pytest.org) (away from
    [nose](https://nose.readthedocs.io)). One thing that may have an
    impact on users is that following the changes made in Version 4.0,
    the PETSc data structures are now much easier, removing a lot of
    unnecessary boilerplate code.

-   May 4, 2021: Long time, no see, but this major release 4.0 marks
    some improvements under the hood:

    -   **Rewrote `mesh` and `particle` data type**:
        Creation of new arrays for each operation is now avoided by
        directly subclassing Numpy\'s `ndarray`. Somewhat faster,
        definitively better, less code, future-proof, but also breaking
        the API. If you use `pySDC` for your project, make
        sure you adapt to the new data types (or don\'t upgrade).
    -   **Faster quadrature**: Thanks to
        [tlunet](https://github.com/tlunet) the computation of the
        weights is now faster and (even) more reliable. No breaking of
        any API here\...
    -   **Bugfixing and version pushes**: The code should run without
        (many) complaints with Python 3.6, 3.7 and potentially above.
        Also, the plotting routines have been adapted to work with
        recent versions of `matplotlib`.

    This is not much (yet) and if it were not for the API changes, this
    would have been a minor release.

-   August 30, 2019: Version 3.1 adds many more examples like the
    nonlinear Schrödinger equation, more on Gray-Scott and in particular
    Allen-Cahn. Those are many implemented using the parallel FFT
    library
    [mpi4pi-fft](https://bitbucket.org/mpi4py/mpi4py-fft/src/master/),
    which can now be used with `pySDC`. There are now 8
    tutorials, where step 7 shows the usage of three external libraries
    with `pySDC`: mpi4py, FEniCS and petsc4py. The MPI controller has
    been improved after performaning a detailed performance analysis
    using [Score-P](https://www.vi-hps.org/projects/score-p/) and
    [Extrae](https://www.vi-hps.org/Tools/Extrae.html). Finally: first
    steps towards error/iteration estimators are taken, too.

-   February 14, 2019: Released version 3 of `pySDC`. This
    release is accompanied by the **ACM TOMS paper**
    ["pySDC -- Prototyping spectral deferred corrections"](https://doi.org/10.1145/3310410).
    It release contains some breaking changes to the API. In detail:

    -   **Dropped Python 2 support**: Starting with this version,
        `pySDC` relies on Python 3. Various incompabilities
        led to inconsistent treatment of dependencies, so that parts of
        the code had to use Python 2 while other relied on Python 3 or
        could do both. We follow [A pledge to migrate to Python
        3](https://python3statement.org/) with this decision, as most
        prominent dependencies of `pySDC` already do.
    -   **Unified controllers**: Instead of providing (and maintaining)
        four different controllers, this release only has one for
        emulated and one for MPI-based time-parallelization
        (`controller_nonMPI` and `controller_MPI`). This should avoid
        further confusion and makes the code easier to maintain. Both
        controllers use the multigrid perspective for the algorithm
        (first exchange data, than compute updates), but the classical
        way of determining when to stop locally (each time-step is
        stopped when ready, if the previous one is ready, too). The
        complete multigrid behavior can be restored using a flag. All
        included projects and tutorials have been adapted to this.
    -   **No more data types in the front-ends**: The redundant use of
        data type specifications in the description dictionaries has
        been removed. Data types are now declared within each problem
        class (more precisely, in the header of the `__init__`-method to
        allow inhertiance). All included projects and tutorials have
        been adapted to this.
    -   **Renewed FEniCS support**: This release revives the deprecated
        [FEniCS](https://fenicsproject.org/) support, now requiring at
        least FEniCS 2018.1. The integration is tested using Travis-CI.
    -   **More consistent handling of local initial conditions**: The
        treatment of `u[0]` and `f[0]` has been fixed and made
        consistent throughout the code.
    -   As usual, many bugs have been discovered and fixed.

-   May 23, 3018: Version 2.4 adds support for
    [petsc4py](https://bitbucket.org/petsc/petsc4py)! You can now use
    [PETSc](http://www.mcs.anl.gov/petsc/) data types
    (`pySDC` ships with DMDA for distributed structured
    grids) and parallel solvers right from your examples and problem
    classes. There is also a new tutorial (7.C) showing this in a bit
    more detail, including communicator splitting for parallelization in
    space and time. Warning: in order to get this to work you need to
    install petsc4py and mpi4py first! Make sure both use MPICH3
    bindings. Downloading `pySDC` from PyPI does not include
    these packages.

-   February 8, 2018: Ever got annoyed at `pySDC`\'s
    incredibly slow setup phase when multiple time-steps are used?
    Version 2.3 changes this by copying the data structure of the first
    step to all other steps using the [dill
    Package](https://pypi.python.org/pypi/dill). Setup times could be
    reduced by 90% and more for certain problems. We also increase the
    speed for certain calculations, in particular for the Penning trap
    example.

-   November 7, 2017: Version 2.2 contains matrix-based versions of
    PFASST within the project `matrixPFASST`. This involved quite a few
    changes in more or less unexpected places, e.g. in the multigrid
    controller and the transfer base class. The impact of these changes
    on other projects should be negligible, though.

-   October 25, 2017: For the [6th Workshop on Parallel-in-Time
    Integration](https://www.ics.usi.ch/index.php/6th-workshop-on-parallel-in-time-methods)
    `pySDC` has been updated to version 2.1. It is now
    available on PyPI - the Python Package Index, see
    <https://pypi.python.org/pypi/pySDC> and can be installed simply by
    using `pip install pySDC`. Naturally, this release contains a lot of
    bugfixes and minor improvements. Most notably, the file structure
    has been changed again to meet the standards for Python packaging
    (at least a bit).

-   November 24, 2016: Released version 2 of `pySDC`. This
    release contains major changes to the code and its structure:

    -   **Complete redesign of code structure**: The `core` part of
        `pySDC` only contains the core modules and classes,
        while `implementations` contains the actual implementations
        necessary to run something. This now includes separate files for
        all collocation classes, as well as a collection of problems,
        transfer classes and so on. Most examples have been ported to
        either `tutorials`, `playgrounds` or `projects`.
    -   **Introduction of tutorials**: We added a tutorial (see below)
        to explain many of pySDC\'s features in a step-by-step fashion.
        We start with a simple spatial discretization and collocation
        formulations and move step by step to SDC, MLSDC and PFASST. All
        tutorials are accompanied by tests.
    -   **New all-inclusive controllers**: Instead of having two PFASST
        controllers which could also do SDC and MLSDC (and more), we now
        have four generic controllers which can do all these methods,
        depending on the input. They are split into two by two class:
        `MPI` and `NonMPI` for real or virtual
        parallelisim as well as `classic` and
        `multigrid` for the standard and multigrid-like
        implementation of PFASST and the likes. Initialization has been
        simplified a lot, too.
    -   **Collocation-based coarsening** As the standard PFASST
        libraries [libpfasst](https://bitbucket.org/memmett/libpfasst)
        and [PFASST++](https://github.com/Parallel-in-Time/PFASST)
        `pySDC` now offers collocation-based coarsening,
        i.e. the number of collocation nodes can be reduced during
        coarsening. Also, time-step coarsening is in preparation, but
        not implemented yet.
    -   **Testing and documentation** The core, implementations and
        plugin packages and their subpackages are fully documented using
        sphinx-apidoc, see below. This documentation as well as this
        website are generated automatically using
        [Travis-CI](https://travis-ci.org/Parallel-in-Time/pySDC). Most
        of the code is supported by tests, mainly realized by using the
        tutorial as the test routines with clearly defined results.
        Also, projects are accompanied by tests.
    -   Further, minor changes:
        -   Switched to more stable barycentric interpolation for the
            quadrature weights
        -   New collocation class: `EquidistantSpline_Right`
            for spline-based quadrature
        -   Collocation tests are realized by generators and not by
            classes
        -   Multi-step SDC (aka single-level PFASST) now works as
            expected
        -   Reworked many of the internal structures for consistency and
            simplicity

:arrow_left: [Back to main page](./README.md)