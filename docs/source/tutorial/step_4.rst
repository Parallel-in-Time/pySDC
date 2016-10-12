Step-4: Multilevel SDC
======================

In this step, we will show how pySDC creates a multilevel hierarchy and
how MLSDC can be run and tested.

Part A: Spatial transfer operators
----------------------------------

For a mjltilevel hierarchy, we need transfer operators. The user, having
knowledge of the data types, will have to provide a
``space_transfer_class`` which deals with restriciton and interpolation
in the spatial dimension. In this part, we simply set up two problems
with two different resolutions and check the order of interpolation (4
in this case).

Important things to note:

-  As for the sweeper and many other things, the user does not have to
   deal with the instantiation of the transfer class, when one of the
   controllers is used. This is for demonstrational purpose only.
-  MLSDC (and PFASST) rely on high-order interpolation in space. When
   using Lagrange-based interpolation, orders above 4-6 are recommended.
   Restriction, however, can be of order 2 and is thus not tested here.

Part B: Multilevel hierarchy
----------------------------

In this example, we demonstrate how the step class creates the
space-time hierarchy dynamically, depending on the description and its
parameters. pySDC supports two different generic coarsening strategies:
coarsening in space and coarsening in the collocation order. To enable
collocation-based coarsening, we simply replace the ``num_nodes``
parameter by a list, where the first entry corresponds to the finest
level. For spatial coarsening, the problem parameter ``nvars`` is
replaced by a list, too. During the step setup, these dictionaries with
lists entries are transformed into lists of dictionaries corresponding
to the levels (3 in this case). A third generic way of creating multiple
levels is to replace an entry in the dscription by a list, e.g. a list
of problem classes. The first entry of each list will always belong to
the finest level.

Important things to note:

-  Not all lists must habe the same length: The longest list defines the
   number of levels and if other lists are shorter, the levels get the
   last entry in these lists (3 nodes on level 1 and 2 in this example).
-  As for most other parameters, ``space_transfer_class`` and
   ``space_transfer_params`` are part of the description of the problem.
-  For advanced users: it is also possible to pass parameters to the
   ``base_transfer`` by specifying ``base_transfer_params`` or even
   replace this by defining ``base_transfer_class`` in the description.

Part C: SDC vs. MLSDC
---------------------

After we have seen how a hierarchy is created, we now run MLSDC and
compare the results with SDC. We create two different descriptions and
controllers and run first SDC and then MLSDC for the simple unforced
heat equation. We see that the results are pretty much the same, while
MLSDC only takes about half as many iterations.

Important things to note:

-  In this case, the number of iterations is halved when using MLSDC.
   This is the best case and in many situations, this cannot be
   achieved. In particular, the interpolation order is crucial.
-  While MLSDC looks less expensive, the number of evaluations of the
   right-hand side of the ODE is basically the same: This is due to the
   fact that after each coarse grid correction (i.e. after the
   interpolation), the right-hand side needs to be re-evaluated on the
   finer level to be prepared for the next sweep. One way of improving
   this is to do the interpolation also in the right-hand side itself.
   This can be achieved by specifying
   ``base_transfer_params['finter] = True`` and by passing it to the
   description. See also Part D.

Part D: MLSDC with particles
----------------------------

For this example, we return to the Boris solver and show how coarsening
can be done beyond degrees of freedom in space or collocation nodes.
Here, we use the setup from ``step_3``, Parts B and C. We run three
different versions of SDC or MLSDC: the standard SDC, the standard MLSDC
and MSDC with interpolation of the right-hand side. For coarsening, we
replace the problem class by a simpler version: the coarse evaluation of
the forces omits the particle-particle interaction and only takes
external forces into account. This is done simply by replacing the
problem class by a list of two problem classes in the description. In
the rsults, we can see that all versions produce more or less the same
energies, where MLSDC without f-interpolation takes about half as many
iterations and with f-interpolation slightly more. We also check the
timings of the three runs: although MLSDC requires much less iterations,
it takes longer to run. This is due to the fact that the right-hand side
of the ODE (i.e. the costly force evaluation) is required after each
interpolation! To this end, we also use f-interpolation, which increases
the iteration counts a bit but leads to a slightly reduced runtime.

Important things to note:

-  Again, the number of MLSDC iterations is highly sensitive to the
   interplay of all the different parameters (number of particles,
   smoothing parameter, number of nodes, residual tolerance etc.). It is
   by far not trivial to get a speedup at all (although a reasonable
   setup has been chosen here).
-  Of course, this type of coarsening can be combined with the generic
   coarsening strategies. Yet, using a reduced number of collocation
   nodes leads to increased iteration counts here.
