Solving differential algebraic equations with SDC
==================================================

This project contains the sweepers, hooks, example problems, plotting and simulation scripts for a Master's thesis investigating the usage of SDC methods to solve differential algebraic equations. 

To run the scripts contained in this project a standard installation of pySDC should suffice. 

Project overview 
--------------------
- hooks
    - | ``HookClass_approx_solution.py``  
      | Simple hook class to read out the approximate solution after each time step.
    - | ``HookClass_error.py``
      | Simple hook class to read out the error between approximate and exact solution after each time step. Note that for some problems the exact solution is only implemented to return initial conditions. In these cases this hook should not be used.

- plotting
    - | ``linear_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on linear axis of the specified parameters.
    - | ``loglog_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on logarithmic axis of the specified parameters. Commonly used to generate convergence plots.

- problems
    - | ``simple_DAE.py`` 
      | A number of simple examples of differential algebraic equations are implemented here: a linear index-2 system with known analytical solution, the 2D pendulum as an index-3 system and a very simple fully implicit index-2 system that is not solvable by most numerical methods for certain values of a parameter.
    - | ``transistor_amplifier.py``
      | A two transistor amplifier model that results in an index-1 differential algebraic system. A nice example of a system resulting from a common real world situation.

- run
    - | ``run_convergence_test.py`` 
      | Script to generate convergence data of applying SDC to the simple linear index-2 differential algebraic system mentioned above. 
    - | ``fully_implicit_dae_playground.py``
      | Testing arena for the fully implicit sweeper. 

- sweepers
    - | ``fully_implicit_DAE.py`` 
      | Sweeper that accepts a fully implicit formulation of a system of differential equations and applies to it a modified version of spectral deferred correction
    - | ``semi_explicit_DAE.py``
      | Sweeper that accepts a semi-explicit formulation of a system of differential algebraic equations and applies to it a modified version of spectral deferred correction

Theoretical details 
----------------------
A fully implicit representation of a system of differential equations takes the form 

.. math::
  F(u'(t), u(t), t) = 0.

A special case of such an implicit differential equation arises when the Jacobian :math:`\partial_{u'}F` is singular. This implies that the derivative of some of the components of :math:`u(t)` do not appear in the system of equations. The system is thus denoted a differential algebraic system of equations. 

Since the derivative :math:`u'(t)` cannot be isolated the Picard formulation used in ordinary SDC cannot be used here. Instead the derivative, henceforth denoted by :math:`U(t)`, is cast as the new unknown solution and the implicit system of equations is written as 

.. math:: 
  F\left(u_0+\int_0^tU(\tau)d\tau, U(t), t\right) = 0.

Based on this equation and an initial approximate solution :math:`tilde{U}`, the following error equation is formed 

.. math:: 
   F\left(u_0+\int_0^t(\tilde{U}(t)+\delta(\tau))d\tau,\;\tilde{U}(t)+\delta(t),\;\right)=0.

This results directly in 

.. math:: 
   F\left(u_0+\int_0^{t_{m+1}}\tilde{U}(\tau)d\tau +\left(\int_0^{t_m} + \int_{t_m}^{t_{m+1}}\right)\delta(\tau)d\tau ,\;\tilde{U}(t_{m+1})+\delta(t_{m+1}),\;t_{m+1}\right)=0

from which the following time marching discretisation becomes obvious

.. math:: 
   F\left(u_0+[\Delta t\mathbf{Q}\tilde{U}]_{m+1} + \sum_{l=1}^{m+1}\Delta t\tilde{\delta}_l,\;\tilde{U}_{m+1}+\tilde{\delta}_{m+1},\;t_{m+1}\right) = 0.

The spectral integration matrix :math:`\mathbf{Q}` is used to approximate the integral of the current approximation :math:`\tilde{U}` and a low order approximation, in this case implicit Euler, is used for the unknown error :math:`\delta(t)`.
Combining each step in the time marching scheme into a vector results in the following matrix formulation 

.. math::
    \mathbf{F}\left(\mathbf{u}_0+\Delta t\mathbf{Q}\tilde{\mathbf{U}} + \Delta t\mathbf{Q}_\Delta\tilde{\mathbf{\delta}},\;\tilde{\mathbf{U}}+\tilde{\mathbf{\delta}},\;\mathbf{t}\right) = \mathbf{0}

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
    \mathbf{F}\left(\mathbf{u}_0+\Delta t(\mathbf{Q}-\mathbf{Q}_\Delta)\tilde{\mathbf{U}} + \Delta t\mathbf{Q}_\Delta(\tilde{\mathbf{U}} + \tilde{\mathbf{\delta}}),\;\tilde{\mathbf{U}}+\tilde{\mathbf{\delta}},\;\mathbf{t}\right) = \mathbf{0}

results in the following iterative scheme

.. math::
    \mathbf{F}\left(\mathbf{u}_0+\Delta t(\mathbf{Q}-\mathbf{Q}_\Delta)\mathbf{U}^{k}+ \Delta t\mathbf{Q}_\Delta\mathbf{U}^{k+1},\;\mathbf{U}^{k+1},\;\mathbf{t}\right) = \mathbf{0}. 

In practice each iteration is carried out line by line and the resulting implicit equation for :math:`U_{m+1}^{k+1}` is solved using the familiar ``scipy.optimize.root()`` function.
