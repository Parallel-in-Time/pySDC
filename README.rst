Welcome to pySDC!
=================

The `pySDC` project is a Python implementation of the spectral deferred correction (SDC) approach and its flavors,
esp. the multilevel extension MLSDC and PFASST. It is intended for rapid prototyping and educational purposes.
New ideas like e.g. sweepers or predictors can be tested and first toy problems can be easily implemented.

Features
--------

- Variants of SDC: explicit, implicit, IMEX, multi-implicit, Verlet, multi-level, diagonal, multi-step
- Variants of PFASST: classic (libpfasst-style) and multigrid, virtual parallel or MPI-based parallel
- 7 tutorials: from setting up a first collocation problem to SDC, PFASST and advanced topics
- Projects: many documented projects with defined and tested outcomes
- Many different examples, collocation types, data types already implemented
- Works with `PETSc <http://www.mcs.anl.gov/petsc/>`_ through `petsc4py <https://bitbucket.org/petsc/petsc4py>`_ and `FEniCS <https://fenicsproject.org/>`_
- Continuous integration via `Travis-CI <https://travis-ci.org/Parallel-in-Time/pySDC>`_
- Fully compatible with Python 2.7 and 3.6 (or higher)


Getting started
---------------

The code is hosted on GitHub, see `https://github.com/Parallel-in-Time/pySDC <https://github.com/Parallel-in-Time/pySDC>`_, and PyPI, see `https://pypi.python.org/pypi/pySDC <https://pypi.python.org/pypi/pySDC>`_.
Either use ``pip install pySDC`` to get the latest stable release including the core dependencies or check out the code on Github.
All package requirements are listed in the files `requirements.txt` (for the core dependencies) and `requirements-optional.txt` for the more advanced features.

To check your installation, run

.. code-block:: bash

   nosetests -v pySDC/tests

Note: When installing both `mpi4py` and `petsc4py`, make sure they use the same MPI installation (e.g. MPICH3).
You can achieve this e.g. by using the `Anaconda distribution <https://www.anaconda.com/distribution/>`_ of Python and then run

.. code-block:: bash

   conda install -c conda-forge petsc4py mpi4py

Most of the code is tested automatically using `Travis-CI <https://travis-ci.org/Parallel-in-Time/pySDC>`_, so a working version of the installation process can always be found in the `install`-block of the `.travis.yml` file.


For more details on `pySDC`, check out `http://www.parallelintime.org/pySDC <http://www.parallelintime.org/pySDC>`_.