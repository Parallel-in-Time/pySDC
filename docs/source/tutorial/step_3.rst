Step-3: Statistics and a new sweeper
====================================

In this step, we will show how to work with the statistics pySDC
generates. We will also introduce a new problem as well as a new
sweeper.

Part A: Getting statistics
--------------------------

In the first part, we run our standard heat equation problem again, but
focus on the ``stats`` dictionary the controller returns in addition to
the final solution. This ``stats`` dictionary is a little bit complex,
as its keys are tuples which contain the process, time, level, iteration
and type of entry for the values. This way, each value has some sort of
time stamp attached to it, so that it is clear when this value was
added. To help dealing with the statistics, we make use of the
``filter_stats`` and ``sort_stats`` routines. The first one filters the
dictionary, following the keys we provide. Here, we would like to have
all residuals logged during time 0.1 (i.e. for all iterations in the
first time step). Analogously, we could ask for all residuals at the
final iteration of each step by calling
``filter_stats(stats, iter=-1, type='residual')``. The second helper
routine converts the filtered or non-filtered dictionary to a listof
tuples, where the first part is the item defined by the parameter
``sortby`` and the second part is the value. Here, we would like to have
a list of iterations and residuals to see how SDC converged over the
iterations.

Important things to note:

-  We now make use of ``controller_parameters``. Here, we use those to
   control the logging verbosity.
-  Admittedly, the ``stats`` dictionary is a complex thing, but it
   allows users to add other details to it without changing its
   definition (see next part).
-  For more details on how the entries of ``stats`` are created, please
   check the ``Hooks`` class.
-  We also use the ``get_list_of_type`` function to show what kind of
   values are registered in the statistics. This can be helpful, if
   users register their own types, see below.

Full code: `tutorial/step_3/A_getting_statistics.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_3/A_getting_statistics.py>`_

.. literalinclude:: ../../../tutorial/step_3/A_getting_statistics.py

Part B: Adding statistics
-------------------------

We now extend the statistics of pySDC by user-defined entries. To make
things much more interesting (and complicated), we introduce a new
problem class (``PenningTrap_3D``) as well a new sweeper
(``boris_2nd_order``). This is accompanied by new data types, namely
``particles`` and ``fields``. Details on the idea of the Boris solver
for particles moving in electromagnetic fields can be found in `this
paper <http://dx.doi.org/10.1016/j.jcp.2015.04.022>`__. Yet, one
important measure for this kind of problem is the total energy of the
system. We would like to compute this after each time step and
therefore, we define a new hook class called ``particle_hooks``. Here,
all necessary computations are done and the value is added to the
statistics for later processing.

Important things to note:

-  In order to extend (and not replace) pySDC's statistics, make sure
   that your custom hook class calls ``super`` each time a function is
   called.
-  User-defined statistics can also be used from within the problem
   class: simply define a problem attribute (e.g. the number of GMRES
   iterations in the spatial solver) and access it during the hook call
   using the level.

Full code: `tutorial/step_3/B_adding_statistics.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_3/B_adding_statistics.py>`_

.. literalinclude:: ../../../tutorial/step_3/B_adding_statistics.py

Part C: Studying collocation node types
---------------------------------------

Having first experience with the Boris-SDC solver, we see that the
energy deviates quite a bit already for this simple setup. We now test
this for three different collocation classes to demonstrate how a
parameter study can be done with pySDC. To this end, we describe the
whole setup up to the parameter we would like to vary. Using a simple
list of parameters, we construct the controller each time using a
different collocation class. While slightly inefficient, this makes sure
that the variables and statistics are reset each time.

Important things to note:

-  Interestingly, the energy does not deviate a lot for Gauss-Lobatto
   and Gauss-Legendre nodes. This is related to their symmetric nature
   (in contrast to Gauss-Radau).
-  Using a lower residual tolerance actually improves the energy
   conservation even further, at least for symmetric collocation nodes.
   Rule of thumb: conservation up to residual tolerance is achieved.
-  Working with multiple ``stats`` dictionaries is not straightforward.
   Yet, putting them into a meta-dictionary is useful (as done here).
   Alternatively, each ``stats`` can be processed on the fly and only
   the relevant information can be stored.

Full code: `tutorial/step_3/C_study_collocations.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_3/C_study_collocations.py>`_

.. literalinclude:: ../../../tutorial/step_3/C_study_collocations.py
