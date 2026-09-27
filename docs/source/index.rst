:html_theme.sidebar_secondary.remove: true

pySDC
=====

.. container:: hero

   .. rst-class:: hero-tagline

   Prototyping spectral deferred corrections, from a single sweep to space-time parallel PFASST.

   :bdg-primary:`SDC` :bdg-primary:`MLSDC` :bdg-primary:`PFASST` :bdg-primary:`ParaDiag`
   :bdg-secondary:`NumPy` :bdg-secondary:`CuPy` :bdg-secondary:`FEniCS` :bdg-secondary:`Firedrake`
   :bdg-secondary:`PETSc` :bdg-secondary:`MPI`

.. code-block:: python

   from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
   from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
   from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

   description = {
       'problem_class': heatNd_forced,
       'problem_params': {'nvars': 128, 'nu': 0.1},
       'sweeper_class': imex_1st_order,
       'sweeper_params': {'num_nodes': 3, 'quad_type': 'RADAU-RIGHT'},
       'level_params': {'dt': 0.1, 'restol': 1e-10},
       'step_params': {'maxiter': 20},
   }
   # four time steps, iterated simultaneously (multi-step SDC); add levels to get PFASST
   controller = controller_nonMPI(num_procs=4, controller_params={}, description=description)
   u0 = controller.MS[0].levels[0].prob.u_exact(0)
   uend, stats = controller.run(u0=u0, t0=0, Tend=0.4)

Try it
------

One time step of the Allen-Cahn equation

.. math::

   \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)

on 128 periodic grid points, with the collocation problem solved by SDC, or by two-level MLSDC with 64 points on the
coarse level. The equation is nonlinear and, for a thin interface ε, stiff; each node is solved with Newton's method.
The plot shows the residual of the collocation problem after each iteration. It runs in your browser, with
`Pyodide <https://pyodide.org>`__; nothing is installed. The code is
`landing_demo.py <https://github.com/Parallel-in-Time/pySDC/blob/master/docs/source/_static/landing_demo.py>`__.

.. landing-demo

Where to go
-----------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em` Overview
      :link: README
      :link-type: doc

      Features, installation, and how to cite pySDC.

   .. grid-item-card:: :octicon:`mortar-board;1.5em` Tutorials
      :link: tutorial/index
      :link-type: doc

      Nine steps, from a first collocation problem to SDC, MLSDC, PFASST and ParaDiag.

   .. grid-item-card:: :octicon:`telescope;1.5em` Projects
      :link: projects/index
      :link-type: doc

      Research built on pySDC: the tested code behind papers, theses and studies.

   .. grid-item-card:: :octicon:`code-square;1.5em` API reference
      :link: api
      :link-type: doc

      Problems, sweepers, controllers, hooks and helpers.

   .. grid-item-card:: :octicon:`book;1.5em` Publications
      :link: publications
      :link-type: doc

      How to cite pySDC, and the research that uses it.

   .. grid-item-card:: :octicon:`git-pull-request;1.5em` Development
      :link: development
      :link-type: doc

      How to contribute, testing, continuous integration and the changelog.

.. toctree::
   :hidden:

   Overview <README>
   tutorial/index
   projects/index
   api
   publications
   development
