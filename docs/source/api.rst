API reference
=============

A run of pySDC is set up with a *description*, a dictionary that names a problem class, a sweeper class and their
parameters, and possibly transfer classes and convergence controllers. A controller then runs it, with hooks, given in
the controller parameters, recording what happens. The tables below list the classes pySDC ships for each of these
roles, with the first sentence of their docstrings; each name links to its full documentation. The complete list of modules is at the end of this page.

.. conf.py replaces the placeholder below with the tables, from the docstrings in pySDC/

.. api-overview

All modules
-----------

.. toctree::
   :maxdepth: 1

   pySDC/pySDC.core
   pySDC/pySDC.implementations
   pySDC/pySDC.helpers
