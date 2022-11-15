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
    - | ``parameter_plot.py``
      | Reads a previously generated data file in `.npy` format and generates a plot on linear axis of the specified parameters.

- problems
    - 
- run
- sweepers
- | ``fully_implicit_dae_playground.py``
  | Testing arena for the fully implicit sweeper. 


Theoretical details 
----------------------



