Welcome to pySDC!
=================

The `pySDC` project is a Python implementation of the spectral deferred correction (SDC) approach and its flavors,
esp. the multilevel extension MLSDC and PFASST. It is intended for rapid prototyping and educational purposes.
New ideas like e.g. sweepers or predictors can be tested and first toy problems can be easily implemented.

The code is hosted on GitHub, see `https://github.com/Parallel-in-Time/pySDC <https://github.com/Parallel-in-Time/pySDC>`_.


News
----

- For the `6th Workshop on Parallel-in-Time Integration <https://www.ics.usi.ch/index.php/6th-workshop-on-parallel-in-time-methods>`_
  `pySDC` has been updated to version 2.1. It is now available on PyPI - the Python Package Index, see `https://pypi.python.org/pypi/pySDC <https://pypi.python.org/pypi/pySDC>`_
  and can be installed simply by using ``pip install pySDC``. Naturally, this release contains a lot of bugfixes and minor improvements.
  Most notably, the file structure has been changed again to meet the standards for Python packaging (at least a bit).

- On November 24, 2016, we released version 2 of `pySDC`. This release contains major changes to the code and its structure:

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


Tutorial
--------

.. include:: ../../pySDC/tutorial/README.rst

.. toctree::
   :maxdepth: 1

   tutorial/step_1.rst
   tutorial/step_2.rst
   tutorial/step_3.rst
   tutorial/step_4.rst
   tutorial/step_5.rst
   tutorial/step_6.rst
   tutorial/step_7.rst

Playgrounds
-----------
.. include:: ../../pySDC/playgrounds/README.rst

Projects
--------

.. include:: ../../pySDC/projects/README.rst

.. toctree::
   :maxdepth: 2

   projects/parallelSDC.rst
   projects/node_failure.rst
   projects/fwsw.rst
   projects/RDC.rst
..   projects/asymp_conv.rst


Tests
-----

.. include:: ../../pySDC/tests/README.rst


API documentation
-----------------

.. include:: ../../pySDC/README_API.rst

.. toctree::
   :maxdepth: 1

   pySDC/core.rst
   pySDC/implementations.rst
   pySDC/helpers.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


