Step-2: Data structures and my first sweeper
============================================

In this step, we will dig a bit deeper into the data structures of
``pySDC`` and work with our first SDC sweeper.

Part A: Step data structure
---------------------------

We start with creating the basic data structure ``pySDC`` is built from:
the ``step``. It represents a single time step with whatever hierarchy
we prescribe (more on that later). The relevant data (e.g. the solution
and right-hand side vectors as well as the problem instances and so on)
are all part of a ``level`` and a set of levels make a ``step``
(together with transfer operators, see teh following tutorials). In this
first example, we simply create a step and test the problem instance of
its only level. This is the same test we ran
`here <../step_1/A_spatial_problem_setup.py>`__.

Important things to note:

-  This part is for demonstration purpose only. Normally, users do not
   have to deal with the internal data structures, see Part C below.
-  The ``description`` dictionary is the central steering tool for
   ``pySDC``.
-  Happily passing around classes instead of instances make life much
   easier, although it is not the best way in terms of programming...

.. include:: doc_step_2_A.rst

Part B: My first sweeper
------------------------

Since we know how to create a step, we now create our first SDC
iteration (by hand, this time). The make use of the IMEX SDC sweeper
``imex_1st_order`` and of the problem class
``HeatEquation_1D_FD_forced``. Also, the data structure for the
right-hand side is now ``rhs_imex_mesh``, since we need implicit and
explicit parts of the right-hand side. The rest is rather
straightforward: we set initial values and times, start by spreading the
data and tehn do the iteration until the maximum number of iterastions
is reached or until the residual is small enough. Yet, this example
sheds light on the key functionalities of the sweeper:
``compute_residual``, ``update_nodes`` and ``compute_end_point``. Also,
the ``step`` and ``level`` structures are explores a bit more deeply,
since we make use of parameters and status objects here.

Important things to note:

-  Again, this part is for demonstration purpose only. Normally, users
   do not have to deal with the internal data structures, see Part C
   below.
-  Note the difference between status and parameter objects: parameters
   are user-defined flags created using the dicitionaries (e.g.
   ``maxiter`` as part of the ``step_params`` in this example), while
   status objects are internal control objects which reflects the
   current status of a level or a step (e.g. ``iter`` or ``time``).
-  The logic required to implement an SDC iteration is simple but also
   rather tedious. This will get worse if you want to deal with more
   than one step or, behold, multiple parallel steps each with a
   space-time hierarchy. Therefore, please proceed quickly to part C!

.. include:: doc_step_2_B.rst

Part C: Using pySDC's frontend
------------------------------

Finally, we arrive at the user-friendliest interface pySDC has to offer.
We use one of the ``controller`` implementations to do the whole
iteration logic for us, namely ``allinclusive_classic_nonMPI``. It can
do SDC, multi-level SDC, multi-step SDC and PFASST, depending on the
input dictionary (more on this in later tutorials). This is the default
controller and does not require ``mpi4py`` to work. It also reflects the
standard PFASST implementation idea, in contrast to the multigrid
counterparts.

Important things to note:

-  By using one of the controllers, the whole code relevant for the user
   is reduced to setting up the ``description`` dictionary, some pre-
   and some post-processing.
-  We make use of ``controller_parameters`` in order to provide logging to file capabilities.
-  In contrast to Part B, we do not have direct access to residuals or
   iteration counts for now. We will deal with these later.
-  This example is the prototype for a user to work with pySDC. Most of
   the logic and most of the data structures are hidden, but all
   relevant parameters are accessable using the ``description``.

.. include:: doc_step_2_C.rst
