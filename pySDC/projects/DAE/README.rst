Solving differential algebraic equations with SDC
==================================================

This project contains the sweepers, hooks, example problems, plotting and simulation scripts for a Master's thesis investigating the usage of SDC methods to solve differential algebraic equations (DAEs). 

To run the scripts contained in this project a standard installation of pySDC should suffice. 

Project overview 
----------------
- misc
    - | ``meshDAE.py``
      | Datatype for semi-explicit DAE problem classes differential and algebraic parts to have a clean treatment.
    - | ``hooksDAE.py``
      | Simple hook classes to read out the approximate solution and error after each time step.
    - | ``problemDAE.py``
      | Parent class for DAE problems containing the method to solve the (non)-linear system at each node/stage.

- plotting
    - | ``linear_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on linear axis of the specified parameters.
    - | ``loglog_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on logarithmic axis of the specified parameters. Commonly used to generate convergence plots.
    - | ``semilogy_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on logarithmic y-axis and linear x-axis. 

- problems
    - | ``discontinousTestDAE.py``
      | Simple nonlinear semi-explicit index-1 DAE with discrete state event whose event time is known.
    - | ``pendulum2D.py``
      | Example of the pendulum described by a semi-implicit DAE of index 3.
    - | ``problematicF.py``
      | Fully-implicit DAE of index 2 which is not solvable for numerically solvable for certain choices of parameter :math:`\eta`.
    - | ``simpleDAE.py`` 
      | Linear semi-explicit index-2 system of Hessenberg form with known analytical solution.
    - | ``synchronousMachine.py`` 
      | Synchronous machine model attached to an infinite bus undergoing torque disturbance test. Results in an index-1 system. 
    - | ``transistorAmplifier.py``
      | A two transistor amplifier model that results in an index-1 differential algebraic system. A nice example of a system resulting from a common real world situation.
    - | ``wscc9BusSystem.py``
      | Large power system test case with three reduced model synchronous machines and nine buses. It is also part of the `PinTSimE project <https://github.com/Parallel-in-Time/pySDC/tree/master/pySDC/projects/PinTSimE>`_.

- run
    - | ``run_convergence_test.py`` 
      | Script to generate convergence data of applying SDC to the simple linear index-2 differential algebraic system mentioned above. 
    - | ``run_iteration_test.py`` 
      | Script to generate data describing behaviour of error and residual of applying SDC to the simple linear index-2 differential algebraic system mentioned above. 
    - | ``fully_implicit_dae_playground.py``
      | Testing arena for the fully implicit sweeper. 
    - | ``synchronous_machine_playground.py``
      | Testing arena for the synchronous machine model.
    - | ``accuracy_check_MPI.py``
      | Script checking the order of accuracy of MPI sweepers for DAEs of different indices.

- sweepers
    - | ``fullyImplicitDAE.py`` 
      | Sweeper that accepts a fully implicit formulation of a system of differential equations and applies to it a modified version of spectral deferred correction
    - | ``semiImplicitDAE.py``
      | SDC sweeper especially for semi-explicit DAEs. This sweeper is based on ideas mentioned in `Huang et al. (2007) <https://www.sciencedirect.com/science/article/abs/pii/S0021999106003147>`_.
    - | ``fullyImplicitDAEMPI.py``
      | MPI version of fully-implicit SDC-DAE sweeper.
    - | ``semiImplicitDAEMPI.py``
      | MPI version of semi-implicit SDC-DAE sweeper.
    - | ``rungeKuttaDAE.py``
      | Runge-Kutta methods that can be used to solve DAEs in pySDC in a fully-implicit description.

- tests
    Here, all tests for the project can be found.
 
Theoretical details 
-------------------
A fully implicit representation of a system of differential equations takes the form 

.. math::
  
  \begin{eqnarray}
     F(u(t), u'(t), t) = 0.
  \end{eqnarray}

A special case of such an implicit differential equation arises when the Jacobian :math:`\partial_{u'}F` is singular. This implies that the derivative of some of the components of :math:`u(t)` do not appear in the system of equations. The system is thus denoted a differential algebraic system of equations. 

Since the derivative :math:`u'(t)` cannot be isolated the Picard formulation used in SDC for ordinary differential equations (ODEs) cannot be used here. Instead the derivative, henceforth denoted by :math:`U(t)`, is cast as the new unknown solution and the implicit system of equations is written as 

.. math::
  
  \begin{eqnarray}
     F\left(u_0+\int_0^tU(\tau)d\tau, U(t), t\right) = 0.
  \end{eqnarray}

The solution :math:`u(t)` can then be recovered using an quadrature step. This approach is also called the *yp-formulation*.

Based on this equation and an initial approximate solution :math:`\tilde{U}`, the following error equation is formed 

.. math::
  
  \begin{eqnarray}
     F\left(u_0+\int_0^t(\tilde{U}(t)+\delta(\tau))d\tau,\;\tilde{U}(t)+\delta(t)\;\right)=0.
  \end{eqnarray}

This results directly in 

.. math::
  
  \begin{eqnarray}
     F\left(u_0+\int_0^{t_{m+1}}\tilde{U}(\tau)d\tau +\left(\int_0^{t_m} + \int_{t_m}^{t_{m+1}}\right)\delta(\tau)d\tau ,\;\tilde{U}(t_{m+1})+\delta(t_{m+1}),\;t_{m+1}\right)=0
  \end{eqnarray}

from which the following time marching discretisation becomes obvious

.. math::
  
  \begin{eqnarray}
     F\left(u_0+[\Delta t\mathbf{Q}\tilde{U}]_{m+1} + \sum_{l=1}^{m+1}\Delta t\tilde{\delta}_l,\;\tilde{U}_{m+1}+\tilde{\delta}_{m+1},\;t_{m+1}\right) = 0.
  \end{eqnarray}

The spectral integration matrix :math:`\mathbf{Q}` is used to approximate the integral of the current approximation :math:`\tilde{U}` and a low order approximation, in this case implicit Euler, is used for the unknown error :math:`\delta(t)`.
Combining each step in the time marching scheme into a vector results in the following matrix formulation 

.. math::
  
  \begin{eqnarray}
     \mathbf{F}\left(\mathbf{u}_0+\Delta t\mathbf{Q}\tilde{\mathbf{U}} + \Delta t\mathbf{Q}_\Delta\tilde{\mathbf{\delta}},\;\tilde{\mathbf{U}}+\tilde{\mathbf{\delta}},\;\mathbf{t}\right) = \mathbf{0}
  \end{eqnarray}

with the integration matrix of the implicit Euler method 

.. math::

  \mathbf{Q}_\Delta=
    \begin{pmatrix}
    \Delta t_1&0&\dots&0&0\\
    \Delta t_1&\Delta t_2&\dots&0&0\\
    .&.&\dots&0&0\\
    \Delta t_1&\Delta t_2&\dots&\Delta t_{M-2}&0\\
    \Delta t_1&\Delta t_2&\dots&\Delta t_{M-2}&\Delta t_{M-1}\\
    \end{pmatrix}

Finally, the iterative nature of the method is made clear by considering that the approximate solution can be updated repeatedly with a :math:`\tilde{\mathbf{\delta}}` that is recalculated after each iteration and using the previously updated solution as the initial condition for the next iteration. In this way, reformulation of the previous equation as 

.. math::
  
  \begin{eqnarray}
     \mathbf{F}\left(\mathbf{u}_0+\Delta t(\mathbf{Q}-\mathbf{Q}_\Delta)\tilde{\mathbf{U}} + \Delta t\mathbf{Q}_\Delta(\tilde{\mathbf{U}} + \tilde{\mathbf{\delta}}),\;\tilde{\mathbf{U}}+\tilde{\mathbf{\delta}},\;\mathbf{t}\right) = \mathbf{0}
  \end{eqnarray}

results in the following iterative scheme

.. math::
  
  \begin{eqnarray}
     \mathbf{F}\left(\mathbf{u}_0+\Delta t(\mathbf{Q}-\mathbf{Q}_\Delta)\mathbf{U}^{k}+ \Delta t\mathbf{Q}_\Delta\mathbf{U}^{k+1},\;\mathbf{U}^{k+1},\;\mathbf{t}\right) = \mathbf{0}. 
  \end{eqnarray}

In practice each iteration is carried out line by line and the resulting implicit equation for :math:`U_{m+1}^{k+1}` is solved using the familiar ``scipy.optimize.root()`` function.

How to implement a DAE problem in pySDC?
----------------------------------------
Different from all other ODE problem classes in ``pySDC`` the DAE problem classes use the *yp-formulation* where the derivative is the unknown and the solution :math:`u` is recovered using quadrature. Interested readers about the different formulations for spectral deferred corrections are referred to `Qu et al. (2015) <https://link.springer.com/article/10.1007/s10915-015-0146-9>`_.

Let us consider the fully-implicit DAE

.. math::

  y' (t) + \eta t z' (t) + (1 + \eta) z (t) &= \cos (t) \\
  y (t) + \eta t z (t) &= \sin (t)

which is of the general form

.. math::
  
  \begin{eqnarray}
     F\left(u (t), u' (t), t\right) = 0
  \end{eqnarray}

.. literalinclude:: ../../../pySDC/projects/DAE/problems/problematicF.py

The imports for the classes ``ProblemDAE`` and ``mesh`` are necessary for implementing this problem.

The problem class inherits from the parent ``ProblemDAE`` that
has the ``solve_system`` method solving the (non)-linear system to find the root, i.e., updating the values of the unknown derivative. All DAE problem classes should therefore inherit from this class.
For this general type of DAEs the datatype ``mesh`` is used here for both, ``u`` and ``f``.
Further, the constructor requires at least the parameter ``newton_tol``, the tolerance passed to the root solver. It is possible to set a default value (which is set to ``1e-8`` in the example above).

**Note:** The name ``newton_tol`` could be confusing. The implicit system is not solved by Newton but rather by a root solver from ``SciPy``. Different quasi-Newton methods can be chosen (see the [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)). Default here is ``'hybr'`` that uses modified Powell hybrid method. To change to solver it is possible to overload the ``solve_system`` method in a new implemented problem class.

Possibly other problem-specific parameters are needed. Our example class also needs a constant ``eta`` set to :math:`1` and storing it as an attribute using ``self._makeAttributeAndRegister('eta', localVars=locals())``.
The system of DAEs consists of two equations, i.e., two unknowns. Thus, the number of variables ``nvars`` needs to be set to :math:`2`.

Implementing this system of equations the problem class also requires the ``eval_f`` method. As it can be seen, the method returns the right-hand side function :math:`F` of the DAE in the way to have a function for which the root is sought.

Since the exact solution is known for this problem, the method ``u_exact`` returns it for each time `t`.

For Runge-Kutta methods, an initial condition for the derivatives at initial time :math:`t_0` :math:`y'(t_0)` and :math:`z'(t_0)` are needed as well. They are implemented using the ``du_exact`` method. 

The second large class of DAEs is the one of semi-explicit form

.. math::

  y' (t) &= f \left(y (t), z (t), t\right) \\
  0 &= g \left(y (t), z (t), t\right)

which is also called a *constrained differential equation*. :math:`y` is the differential variable and :math:`z` denotes the algebraic variable since no corresponding integration is in the problem.
We want to implement such an equation and consider the example

.. math::

  u_1' (t) &= (\alpha - \frac{1}{2 - t}) u_1 (t) + (2 - t) \alpha z (t) + \frac{3 - t}{2 - t}, \\
  u_2' (t) &= \frac{1 - \alpha}{t - 2} u_1 (t) - u_2 (t) + (\alpha - 1) z (t) + 2 e^t, \\
  0 &= (t + 2) u_1 (t) + (t^2 - 4) u_2 (t) - (t^2 + t - 2) e^t.

This example has two differential variables :math:`u_1`, :math:`u_2` (two differential equations) and one algebraic variable :math:`z` (thus one algebraic equation).
In ``pySDC`` defining a problem class for semi-explicit DAEs is slightly different to those of fully-implicit form. Additionally to ``numpy`` for the example the imports for the classes ``ProblemDAE`` and ``MeshDAE`` are needed.

.. literalinclude:: ../../../pySDC/projects/DAE/problems/simpleDAE.py

This problem class inherits again from ``ProblemDAE``. In contrast, for the solution ``u`` and the right-hand side of the ``f``
a different datatype ``MeshDAE`` is used that allows to separate between the differential variables and the algebraic variables as well
as for the equations. The tolerance for the root solver is passed with a default value of ``1e-10`` and the number of unknowns is :math:`3`, i.e., ``nvars=3``.
The problem-specific parameter ``a`` has a default value of ``10.0``.

In the ``eval_f`` method the equations and the variables are now separated using the components of the ``MeshDAE``. Recall that ``eval_f`` returns the right-hand side function so that we have a root problem. However, for this semi-explicit DAE this is not the case, but we can change that by rewriting the system to

.. math::

  0 &= f \left(y (t), z (t), t\right) - y' (t) \\
  0 &= g \left(y (t), z (t), t\right).

In the example above the differential variables are :math:`u_1` and :math:`u_2` which can be accessed using ``u.diff[0]`` and ``u.diff[1]``.
The algebraic variable :math:`z` is stored in ``u.alg[0]``. The corresponding derivatives for :math:`u_1` and :math:`u_2` are stored in ``du.diff[0]`` and ``du.diff[1]``.
It is also possible to separate the differential and algebraic equations by assigning the corresponding equations to ``f.diff[0]`` and ``f.diff[1]``, and ``f.alg[0]``, respectively.

In the same way the method ``u_exact`` to access the exact solution can be implemented.

This example class also have an ``du_exact`` method to implement initial conditions for :math:`u_1'(t_0)`, :math:`u_2'(t_0)`, and :math:`z'(t_0)`.
