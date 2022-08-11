from __future__ import division

import dill
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.projects.soft_failure.FaultHooks import fault_hook
from pySDC.projects.soft_failure.implicit_sweeper_faults import implicit_sweeper_faults
from pySDC.projects.soft_failure.visualization_helper import (
    show_residual_across_simulation,
    show_min_max_residual_across_simulation,
    show_iter_hist,
)


def diffusion_setup():
    """
    Setup routine for diffusion test
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['detector_threshold'] = 1e-10

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 127  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = implicit_sweeper_faults  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def reaction_setup():
    """
    Setup routine for diffusion-reaction test with Newton solver
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['detector_threshold'] = 1e-10

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1.0
    problem_params['lambda0'] = 2.0
    problem_params['newton_maxiter'] = 20
    problem_params['newton_tol'] = 1e-10
    problem_params['stop_at_nan'] = False
    problem_params['interval'] = (-5, 5)
    problem_params['nvars'] = 127

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = generalized_fisher  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = implicit_sweeper_faults  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def vanderpol_setup():
    """
    Van der Pol's oscillator
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['detector_threshold'] = 1e-10

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-10
    problem_params['newton_maxiter'] = 50
    problem_params['stop_at_nan'] = False
    problem_params['mu'] = 18
    problem_params['u0'] = (1.0, 0.0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = implicit_sweeper_faults
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def run_clean_simulations(type=None):
    """
    A simple code to run fault-free simulations

    Args:
        type (str): setup type
        f: file handler
    """

    if type == 'diffusion':
        description, controller_params = diffusion_setup()
    elif type == 'reaction':
        description, controller_params = reaction_setup()
    elif type == 'vanderpol':
        description, controller_params = vanderpol_setup()
    else:
        raise ValueError('No valid setup type provided, aborting..')

    # set time parameters
    t0 = 0.0
    Tend = description['level_params']['dt']

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # this is where the iteration is happening
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # print('This clean run took %s iterations!' % iter_counts[0][1])

    return iter_counts[0][1]


def run_faulty_simulations(type=None, niters=None, cwd=''):
    """
    A simple program to run faulty simulations

    Args:
        type (str): setup type
        niters (int): number of iterations in clean run
        f: file handler
        cwd (str): current workind directory
    """

    if type == 'diffusion':
        description, controller_params = diffusion_setup()
    elif type == 'reaction':
        description, controller_params = reaction_setup()
    elif type == 'vanderpol':
        description, controller_params = vanderpol_setup()
    else:
        raise ValueError('No valid setup type provided, aborting..')

    # set time parameters
    t0 = 0.0
    Tend = description['level_params']['dt']

    filehandle_injections = open(cwd + 'data/dump_injections_' + type + '.txt', 'w')

    controller_params['hook_class'] = fault_hook
    description['sweeper_params']['allow_fault_correction'] = True
    description['sweeper_params']['dump_injections_filehandle'] = filehandle_injections
    description['sweeper_params']['niters'] = niters

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # number of runs
    nruns = 500
    results = []
    for _ in range(nruns):
        # this is where the iteration is happening
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        results.append(stats)

    filehandle_injections.close()

    dill.dump(results, open(cwd + "data/results_" + type + ".pkl", "wb"))


def process_statistics(type=None, cwd=''):

    results = dill.load(open(cwd + "data/results_" + type + ".pkl", "rb"))

    # get minimal length of residual vector
    minlen = 1000
    nruns = 0
    for stats in results:
        residuals = get_sorted(stats, type='residual_post_iteration', sortby='iter')
        minlen = min(minlen, len(residuals))
        nruns += 1

    # initialize minimal residual vector
    minres = np.zeros(minlen)
    minres[:] = 1000
    # initialize maximal residual vector
    maxres = np.zeros(minlen)
    # initialize mean residual vector
    meanres = np.zeros(minlen)
    # initialize median residual vector
    medianres = np.zeros(minlen)
    # initialize helper list
    median_list = [[] for _ in range(minlen)]

    for stats in results:
        # Some black magic to extract fault stats out of monstrous stats object
        # fault_stats = get_sorted(stats, type='fault_stats', sortby='type')[0][1]
        # Some black magic to extract residuals dependent on iteration out of monstrous stats object
        residuals_iter = get_sorted(stats, type='residual_post_iteration', sortby='iter')
        # extract residuals ouf of residuals_iter
        residuals = np.array([item[1] for item in residuals_iter])

        # calculate minimal, maximal, mean residual vectors
        for i in range(minlen):
            if np.isnan(residuals[i]) or np.isinf(residuals[i]):
                residuals[i] = 1000
            minres[i] = min(minres[i], residuals[i])
            maxres[i] = max(maxres[i], residuals[i])
            meanres[i] += residuals[i]
            median_list[i].append(residuals[i])

        # Example output of what we now can do
        # print(fault_stats.nfaults_injected_u, fault_stats.nfaults_injected_f, fault_stats.nfaults_detected,
        #       fault_stats.nfalse_positives, fault_stats.nfalse_positives_in_correction,
        #       fault_stats.nfaults_missed, fault_stats.nclean_steps)

        # print()

    # call helper routine to produce residual plot
    # fname = 'residuals.png'
    fname = cwd + 'data/' + type + '_' + str(nruns) + '_' + 'runs' + '_' + 'residuals.png'
    show_residual_across_simulation(stats=stats, fname=fname)
    meanres /= nruns
    # print(minres)
    # print(maxres)
    # print(meanres)
    # calculate median residual vector
    for i in range(minlen):
        medianres[i] = np.median(median_list[i])
    # print(median_list)
    # print(medianres)
    # call helper routine to produce residual plot of minres, maxres, meanres and medianres
    # fname = 'min_max_residuals.png'
    fname = cwd + 'data/' + type + '_' + str(nruns) + '_' + 'runs' + '_' + 'min_max_residuals.png'
    show_min_max_residual_across_simulation(
        fname=fname, minres=minres, maxres=maxres, meanres=meanres, medianres=medianres, maxiter=minlen
    )

    # calculate maximum number of iterations per test run
    maxiter = []
    for stats in results:
        residuals = get_sorted(stats, type='residual_post_iteration', sortby='iter')
        maxiters = max(np.array([item[0] for item in residuals]))
        maxiter.append(maxiters)
    # print(maxiter)
    # call helper routine to produce histogram of maxiter
    # fname = 'iter_hist.png'
    fname = cwd + 'data/' + type + '_' + str(nruns) + '_' + 'runs' + '_' + 'iter_hist.png'
    show_iter_hist(fname=fname, maxiter=maxiter, nruns=nruns)

    # initialize sum of nfaults_detected
    nfd = 0
    # initialize sum of nfalse_positives
    nfp = 0
    # initialize sum of nfaults_missed
    nfm = 0
    # initialize sum of nfalse_positives_in_correction
    nfpc = 0
    # calculate sum of nfaults_detected, sum of nfalse_positives, sum of nfaults_missed
    for stats in results:
        # Some black magic to extract fault stats out of monstrous stats object
        fault_stats = get_sorted(stats, type='fault_stats', sortby='type')[0][1]
        nfd += fault_stats.nfaults_detected
        nfp += fault_stats.nfalse_positives
        nfm += fault_stats.nfaults_missed
        nfpc += fault_stats.nfalse_positives_in_correction

    g = open(cwd + 'data/' + type + '_' + str(nruns) + '_' + 'runs' + '_' + 'Statistics.txt', 'w')
    out = 'Type: ' + type + ' ' + str(nruns) + ' runs'
    g.write(out + '\n')
    # detector metrics (Sloan, Kumar, Bronevetsky 2012)
    # nfaults_detected
    out = 'true positives: ' + str(nfd)
    g.write(out + '\n')
    # nfaults_positives
    out = 'false positives: ' + str(nfp)
    g.write(out + '\n')
    # nfaults_missed
    out = 'false negatives: ' + str(nfm)
    g.write(out + '\n')
    # nfalse_positives_in_correction
    out = 'false positives in correction: ' + str(nfpc)
    g.write(out + '\n')
    # F-Score
    f_score = 2 * nfd / (2 * nfd + nfp + nfm)
    out = 'F-Score: ' + str(f_score)
    g.write(out + '\n')
    # false positive rate (FPR)
    fpr = nfp / (nfd + nfp)
    out = 'False positive rate: ' + str(fpr)
    g.write(out + '\n')
    # true positive rate (TPR)
    tpr = nfd / (nfd + nfp)
    out = 'True positive rate: ' + str(tpr)
    g.write(out + '\n')
    g.close()


def main():

    # type = 'diffusion'
    # niters = run_clean_simulations(type=type)
    # run_faulty_simulations(type=type, niters=niters)
    # process_statistics(type=type)

    # type = 'reaction'
    # niters = run_clean_simulations(type=type)
    # run_faulty_simulations(type=type, niters=niters)
    # process_statistics(type=type)

    type = 'vanderpol'
    niters = run_clean_simulations(type=type)
    run_faulty_simulations(type=type, niters=niters)
    process_statistics(type=type)


if __name__ == "__main__":
    main()
