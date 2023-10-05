# Spectral Deferred Corrections with Discontinuity Handling for Dynamic Power System Simulation

The Python file `paper_plots.py` creates all the plots contained in the publication:

**Title:** Spectral Deferred Corrections with Discontinuity Handling for Dynamic Power System Simulation

**Authors:** Junjie Zhang, Lisa Wimmer, Robert Speck, Matthias Bolten, Kyrill Ho, and Andrea Benigni

**Conference:** [![PSCC 2024](http://http://pscc2024.fr/)](http://http://pscc2024.fr/)

Current status of the submission: ***submitted***

## Plots for the discontinuous test DAE
In order to reproduce the plots for the discontinuous test DAE, the following setting is used: The test DAE `DiscontinuousTestDAE`
is simulated over the time domain with `t0=3.0` and `Tend=5.4` for different step sizes `dt_list = [1 / (2 ** m) for m in range(2, 9)]`. The fully implicit SDC-DAE sweeper `fully_implicit_DAE` solves the problem for different number of collocation
nodes `nnodes=[2, 3, 4, 5]` at Radau IIa nodes `quad_type='RADAU-RIGHT` with LU preconditioner `QI='LU'` using `tol_hybr=1e-6` to solve the nonlinear system.
SDC terminates either the maximum number of iterations `maxiter=45` or the residual tolerance `restol=1e-13` is satisfied.
For event detection, if an event is found the step sizes will be adapted using the factor `alpha=0.95`. A founded event should satisfy the tolerance `epsilon_SE=1e-10`. 

Then, executing `make_plots_for_test_DAE()` creates the plots, where functions in the script

    - Fig. 1: `plot_functions_over_time()` for `dt_fix=1 / (2 ** 7)`,
    - Fig. 2: `plot_state_function_detection()`,
    - Fig. 3: `plot_event_time_error_before_restarts()` for `dt_fix=1 / (2 ** 7)`

are used. Here, the routine contains additional functions `plot_error_norm()` and `plot_event_time_error()` to create further plots
not used in the publication. The interested applicant is referred to also consider these ones.

## Plots for the WSCC 9-bus test case
To reproduce the plots for the WSCC 9-bus system, enable the function `make_plots_for_WSCC9_test_case()`. It is recomended to execute the script generating the plots for the WSCC 9-bus test case on a cluster, since the execution takes several hours. Use the following setup: The DAE `WSCC9BusSystem` is simulated over the time domain with `t0=0.0` and `Tend=0.7` for different step sizes `dt_list = [1 / (2 ** m) for m in range(5, 11)]`. The fully implicit SDC-DAE sweeper `fully_implicit_DAE` solves the problem for different number of collocation nodes `nnodes=[2, 3, 4, 5]` at Radau IIa nodes `quad_type='RADAU-RIGHT` with LU preconditioner `QI='LU'` using `tol_hybr=1e-10` to solve the nonlinear system.
SDC terminates either the maximum number of iterations `maxiter=50` or the residual tolerance `restol=5e-13` is satisfied. For event detection, if an event is found the step sizes will be adapted using the factor `alpha=0.95`. A found event should satisfy the tolerance `epsilon_SE=1e-10`. 

Then, executing `make_plots_for_WSCC9_test_case()` creates the plots, where functions in the script

    - Fig. 5: `plot_state_function_detection()`,
    - Fig. 6: `plot_functions_over_time()` for `dt_fix=1 / (2 ** 8)`

are used.

It is important to note that the `SwitchEstimator` is under ongoing development. Using the newest version of `pySDC` could therefore lead to different results, and hence to different plots. For reproducing the plots in the paper, the commit (add commit here) should be used.