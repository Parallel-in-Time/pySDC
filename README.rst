Welcome to pySDC!
=================

The `pySDC` project is a Python implementation of the spectral deferred correction (SDC) approach and its flavors,
esp. the multilevel extension MLSDC and PFASST. It is intended for rapid prototyping and educational purposes.
New ideas like e.g. sweepers or predictors can be tested and first toy problems can be easily implemented.

Features
--------

- Variants of SDC: explicit, implicit, IMEX, multi-implicit, Verlet, multi-level, diagonal, multi-step
- Variants of PFASST: virtual parallel or MPI-based parallel, classical of multigrid perspective
- 7 tutorials: from setting up a first collocation problem to SDC, PFASST and advanced topics
- Projects: many documented projects with defined and tested outcomes
- Many different examples, collocation types, data types already implemented
- Works with `PETSc <http://www.mcs.anl.gov/petsc/>`_ through `petsc4py <https://bitbucket.org/petsc/petsc4py>`_ and `FEniCS <https://fenicsproject.org/>`_
- Continuous integration via `Travis-CI <https://travis-ci.org/Parallel-in-Time/pySDC>`_
- Fully compatible with 3.6 (or higher)


Getting started
---------------

The code is hosted on GitHub, see `https://github.com/Parallel-in-Time/pySDC <https://github.com/Parallel-in-Time/pySDC>`_, and PyPI, see `https://pypi.python.org/pypi/pySDC <https://pypi.python.org/pypi/pySDC>`_.
Either use ``pip install pySDC`` to get the latest stable release including the core dependencies or check out the code on Github.
Note that using ``pip install pySDC`` or ``python setup.py install`` will only install the core dependencies, omitting `mpi4py` and `petsc4py` (see below).
All package requirements are listed in the files `requirements.txt <https://github.com/Parallel-in-Time/pySDC/blob/master/requirements.txt>`_ .

To check your installation, run

.. code-block:: bash

   nosetests -v pySDC/tests

You may need to update your ``PYTHONPATH`` by running

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:../../..

in particular if you want to run any of the playgrounds, projects or tutorials.
All ``import`` statements there assume that the `pySDC`'s base directory is part of ``PYTHONPATH``.

Note: When installing both `mpi4py` and `petsc4py`, make sure they use the same MPI installation (e.g. MPICH3).
You can achieve this e.g. by using the `Anaconda distribution <https://www.anaconda.com/distribution/>`_ of Python and then run

.. code-block:: bash

   conda install -c conda-forge mpich petsc4py mpi4py

Most of the code is tested automatically using `Travis-CI <https://travis-ci.org/Parallel-in-Time/pySDC>`_, so a working version of the installation process can always be found in the `install`-block of the `.travis.yml <https://github.com/Parallel-in-Time/pySDC/blob/master/.travis.yml>`_ file.

For many examples, `LaTeX` is used for the plots, i.e. a decent installation of this is needed in order to run the tests.
When using `FEniCS` or `petsc4py`, a C++ compiler is required (although installation may go through at first).

For more details on `pySDC`, check out `http://www.parallel-in-time.org/pySDC <http://www.parallel-in-time.org/pySDC>`_.
