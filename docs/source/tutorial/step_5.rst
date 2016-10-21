Step-5: PFASST
==============

In this step, we will show how pySDC can do PFASST runs (virtually parallel for now).

Part A: Multistep multilevel hierarchy
--------------------------------------

In this first part, we create a controller and demonstrate how pySDC's data structures represent multiple time-steps.
While for SDC the ``step`` data structure is the key part, we now habe simply a list of steps, bundled in the ``MS`` attribute of the controller.
The nice thing about going form MLSDC to PFASST is that only the number of processes in the ``num_procs`` variable has to be changed.
This way the controller knows that multiple steps have to be computed in parallel.

Important things to note:

- To avoid the tedious installation of mpi4py and to have full access to all data at all times, the controllers with the ``_nonMPI`` suffix only emulate parallelism.
  The algorithm is the same, but the steps are performed serially. Using ``MPI`` controllers allow for real parallelism and should yield the same results (see next tutorial step).
- While in principle all steps can have a different number of levels, the controllers implemented so far assume that the number of levels is constant.
  Also, the instantiation of (the list of) steps via the controllers is implemented only for this case. Yet, pySDC's data structures in principle allow for different approaches.

Full code: `tutorial/step_5/A_multistep_multilevel_hierarchy.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_5/A_multistep_multilevel_hierarchy.py>`_

.. literalinclude:: ../../../tutorial/step_5/A_multistep_multilevel_hierarchy.py

Results:

.. literalinclude:: ../../../step_5_A_out.txt

Part B: My first PFASST run
---------------------------

After we have created our multistep-multilevel hierarchy, we are now ready to run PFASST.
We choose the simple heat equation for our first test.
One of the most important characteristic of a parallel-in-time algorithm is its behavior for increasing number of parallel time-steps for a given problem (i.e. with ``dt`` and ``Tend`` fixed).
Therefore, we loop over the number of parallel time-steps in this eample to see how PFASST performs for 1, 2, ..., 16 parallel steps.
We compute and check the error as well as multiple statistical quantaties, e.g. the mean number of iterations, the range of iterations counts and so on.
We see that PFASST performs very well in this case, the iteration counts do not increase significantly.

- In the IMEX sweeper, we can activate the LU-trick for the implicit part by specifying ``do_LU`` as ``True``. For stiff parabolic problems with Gauss-Radau nodes, this is usually a very good idea!
- As usual for MLSDC and PFASST, the success depends heavily on the choice of parameters.
  Making the problem more complicated, less/more stiff, changing the order of the spatial interpolation etc. can give completely different results.

Full code: `tutorial/step_5/B_my_first_PFASST_run.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_5/B_my_first_PFASST_run.py>`_

.. literalinclude:: ../../../tutorial/step_5/B_my_first_PFASST_run.py

Results:

.. literalinclude:: ../../../step_5_B_out.txt


Part C: Advection and PFASST
----------------------------

We saw in the last part that PFASST does perform very well for certain parabolic problems. Now, we test PFASST for an advection test case to see how things go then.
The basic set up is the same, but now using only an implicit sweeper and periodic boundary conditions.
To make things more interesting, we choose two different sweepers: the LU-trick as well as the implicit Euler and check how these are performing for this kind of problem.
We see that in contrast to the parabolic problem, the iteration counts actually increase significantly, if more parallel time-steps are computed.
Again, this heavily depends on the actual problem under consideration, but it is a typical behavior of parallel-in-time algorithms of this type.

Important things to note:

- The setup is actually periodic in time as well! At ``Tend = 1`` the exact solution looks exactly like the initial condition.
- The ``generic_implicit`` sweeper allows the user to change the preconditioner, named ``QI``. To get the standard implicit Euler scheme, choose ``IE``, while for the LU-trick, choose ``LU``.
  More choices have been implemented in ``pySDC.plugins.sweeper_helper.get_Qd``.

Full code: `tutorial/step_5/C_advection_and_PFASST.py <https://github.com/Parallel-in-Time/pySDC/blob/pySDC_v2/tutorial/step_5/C_advection_and_PFASST.py>`_

.. literalinclude:: ../../../tutorial/step_5/C_advection_and_PFASST.py

Results:

.. literalinclude:: ../../../step_5_C_out.txt
