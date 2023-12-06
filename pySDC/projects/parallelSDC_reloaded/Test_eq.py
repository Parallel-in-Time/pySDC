import os
import pickle
from collections import namedtuple

import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

def main():

    qd_list = ['IEpar', 'MIN-SR-NS', 'MIN-SR-S', 'FLEX-MIN-1', 'MIN3']

    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = 1

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 4
    sweeper_params['initial_guess'] = 'zero'

    step_params = dict()
    step_params['maxiter'] = 100

    controller_params = dict()
    controller_params['logger_level'] = 30

    problem_params = dict()
    problem_params['lambdas'] = np.array([10j])
    problem_params['u0'] = 1.0

    description = dict()
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['level_params'] = level_params
    description['sweeper_class'] = generic_implicit
    description['step_params'] = step_params

    # get a solution of the collocation problem
    sweeper_params['QI'] = 'LU'
    description['sweeper_params'] = sweeper_params
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    uinit = controller.MS[0].levels[0].prob.u_exact(0)
    coll_exact, _ = controller.run(u0=uinit, t0=0, Tend=level_params['dt'])

    level_params['nsweeps'] = 1
    description['level_params'] = level_params

    exact_dt = controller.MS[0].levels[0].prob.u_exact(level_params['dt'])
    err = np.linalg.norm(exact_dt - coll_exact, np.inf)
    print('M = ', sweeper_params['num_nodes'])
    print('error exact - approx = ', err, '\n')

    # loop over all Q-delta matrix types
    for qd_type in qd_list:

        sweeper_params['QI'] = qd_type
        description['sweeper_params'] = sweeper_params

        step_params['maxiter'] = sweeper_params['num_nodes']
        description['step_params'] = step_params

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        uinit = controller.MS[0].levels[0].prob.u_exact(0)

        # call main function to get things done...
        uend, _ = controller.run(u0=uinit, t0=0, Tend=level_params['dt'])
        err = np.linalg.norm(uend - coll_exact, np.inf) / np.linalg.norm(coll_exact, np.inf)
        print('Qd = {:12}, after {} iter, error = {}'.format(qd_type, step_params['maxiter'], err))


if __name__ == "__main__":
    main()
