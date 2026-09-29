Time-dependent boundary conditions, spectrally
==============================================

The heat equation :math:`u_t = \nu u_{xx}` on :math:`(-1, 1)` with Dirichlet data
:math:`a\cos t` and :math:`b\cos t`, discretized with the ultraspherical method of
``Heat1DUltraspherical``. The boundary conditions sit in the last two rows of the linear system, so
changing them per collocation node is one assignment. There is no spatial error floor above roundoff,
which makes this a cleaner testbed than the FEniCS sibling in ``playgrounds/FEniCS/order_reduction``
or the Taylor-Green study in ``projects/StroemungsRaum``.

Idea and first version by Thomas Baumann, pull request #634.

Three ways of imposing the data (``bc_mode``):

- ``pointwise``: each stage takes :math:`g(\tau_m)`.
- ``lifted``: solve for :math:`v = u - E` with :math:`E` the linear interpolant of the data, which has
  homogeneous data and the forcing :math:`-E_t`.
- ``differentiated``: each stage takes :math:`g(t_0) + \Delta t \sum_j Q_{mj}\dot g(\tau_j)`; needs the
  ``generic_implicit_diffbc`` sweeper.

Run ``python run_time_dep_BCs.py`` (a few minutes). Observed orders at the finest step sizes, converged
collocation, RADAU-RIGHT, max norm on the grid:

===  ======  ========  =========  ======  ==============
M    design  constant  pointwise  lifted  differentiated
===  ======  ========  =========  ======  ==============
3    5       4.98      4.04       4.80    4.78
4    7       6.95      4.96       6.09    6.10
===  ======  ========  =========  ======  ==============

Pointwise data costs the method its order: it falls to :math:`M+1`. Both remedies gain one order, to
:math:`M+2`, which restores the design order for :math:`M \le 3` but not beyond. At :math:`M = 4` this is a
ceiling, not pre-asymptotics: in the same range of step sizes, constant data shows order 7.

Lifting and differentiating give the same numbers because here they are the same method. The lift is
linear in :math:`x`, so :math:`E_{xx} = 0`, and the lifted stages equal the differentiated ones plus the
quadrature error of :math:`E` at each node. At the end of a step that is the full Radau quadrature,
:math:`O(\Delta t^{2M})`, which is below everything measured here. For a lift that the spatial operator
does not annihilate, the two would differ.
