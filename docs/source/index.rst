.. pySDC documentation master file, created by
   sphinx-quickstart on Tue Oct 11 15:58:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pySDC's documentation!
=================================

The pySDC project is a Python implementation of the spectral deferred correction (SDC) approach and its flavors,
esp. the multilevel extension MLSDC and PFASST.
It is intended for rapid prototyping and educational purposes.
New ideas like e.g. sweepers or predictors can be tested and first toy problems can be easily implemented.

See `test coverage results <test_coverage/index.html>`_


Tutorial
--------

.. include:: ../../tutorial/README.rst

.. toctree::
   :maxdepth: 1

   tutorial/step_1.rst
   tutorial/step_2.rst
   tutorial/step_3.rst
   tutorial/step_4.rst
   tutorial/step_5.rst
   tutorial/step_6.rst
   tutorial/step_7.rst

Projects
--------

.. toctree::
   :maxdepth: 2

   projects/parallelSDC.rst
   projects/node_failure.rst
   projects/fwsw.rst

Playgrounds
-----------
.. include:: ../../playgrounds/README.rst

API documentation
-----------------

.. include:: ../../pySDC/README.rst

.. toctree::
   :maxdepth: 3

   pySDC/pySDC.core.rst
   pySDC/pySDC.implementations.rst
   pySDC/pySDC.helpers.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

