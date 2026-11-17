"""
Running SDC-SBB and the classical schemes on an ensemble of Brownian paths, and measuring their strong and weak errors.

All methods of a study see the same paths, and the errors are measured against the exact solution for those paths:

- strong error: mean over the samples of the absolute error at the end time, per component
- weak error: absolute difference between the sample means of the numerical and the exact solution at the end time,
  per component (the paper's proxy for the error in the expectation, free of Monte Carlo noise)

Work is the number of evaluations of drift, diffusion and (for Milstein) derivative of the diffusion per sample.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.SDC4SDEs.baselines import euler_maruyama, milstein
from pySDC.projects.SDC4SDEs.sweeper import SBBSweeper


def solve_sdc(problem_class, problem_params, dt, num_nodes, num_sweeps, initial_guess='euler', Tend=1.0):
    """
    Solve with SDC-SBB on Lobatto nodes, with a fixed number of sweeps per step.

    Args:
        problem_class: subclass of SDEProblem
        problem_params (dict): its parameters, with the Brownian paths
        dt (float): step size
        num_nodes (int): number of Lobatto nodes
        num_sweeps (int): number of sweeps after the initial guess
        initial_guess (str): 'spread', 'euler' or 'euler_maruyama'
        Tend (float): end time

    Returns:
        numpy.ndarray: solution at Tend, shape (n_samples, ndim)
        SDEProblem: the problem, with its work counters
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': SBBSweeper,
        'sweeper_params': {
            'num_nodes': num_nodes,
            'quad_type': 'LOBATTO',
            'initial_guess': initial_guess,
            # the residual is not needed with a fixed number of sweeps, and would cost an integration per sweep
            'skip_residual_computation': ('IT_CHECK',),
        },
        'level_params': {'dt': dt, 'restol': -1, 'residual_type': 'last_abs'},
        'step_params': {'maxiter': num_sweeps},
    }
    controller = controller_nonMPI(1, {'logger_level': 30, 'dump_setup': False, 'hook_class': []}, description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return np.asarray(uend), P


def solve(method, problem_class, problem_params, dt, Tend=1.0):
    """
    Solve with one of the methods.

    Args:
        method: 'EM', 'Milstein', or a dict of keyword arguments of :func:`solve_sdc` (num_nodes, num_sweeps and
            optionally initial_guess)
        problem_class: subclass of SDEProblem
        problem_params (dict): its parameters, with the Brownian paths
        dt (float): step size
        Tend (float): end time

    Returns:
        numpy.ndarray: solution at Tend, shape (n_samples, ndim)
        SDEProblem: the problem, with its work counters
    """
    if isinstance(method, dict):
        return solve_sdc(problem_class, problem_params, dt, Tend=Tend, **method)
    P = problem_class(**problem_params)
    return np.asarray({'EM': euler_maruyama, 'Milstein': milstein}[method](P, dt, Tend)), P


def errors(u, P, Tend=1.0):
    """Strong and weak error per component, see the module docstring."""
    exact = np.asarray(P.u_exact(Tend))
    return np.mean(np.abs(u - exact), axis=0), np.abs(np.mean(u, axis=0) - np.mean(exact, axis=0))


def convergence(method, problem_class, problem_params, dts, Tend=1.0):
    """
    Errors and work of a method for a range of step sizes.

    Returns:
        dict: arrays 'dt', 'strong' and 'weak' (shape (len(dts), ndim)), and 'work' (per sample)
    """
    result = {'dt': np.asarray(dts), 'strong': [], 'weak': [], 'work': []}
    for dt in dts:
        u, P = solve(method, problem_class, problem_params, dt, Tend)
        strong, weak = errors(u, P, Tend)
        result['strong'].append(strong)
        result['weak'].append(weak)
        result['work'].append(sum(counter.niter for counter in P.work_counters.values()))
    return {key: np.asarray(value) for key, value in result.items()}


def order(dts, errors):
    """Slope of a least-squares fit of log(errors) against log(dts)."""
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]
