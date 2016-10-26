import pickle
from collections import namedtuple

import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.Van_der_Pol_oscillator import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.plugins.stats_helper import filter_stats, sort_stats

ID = namedtuple('ID', ['setup', 'qd_type', 'param'])


def main():

    # initialize level parameters (part I)
    level_params = dict()
    level_params['restol'] = 1E-10

    # initialize sweeper parameters (part I)
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # set up list of Q-delta types and setups
    qd_list = ['LU', 'IE', 'IEpar', 'Qpar', 'PIC']
    setup_list = [('heat', 63, [10.0 ** i for i in range(-3, 3)]),
                  ('advection', 64, [10.0 ** i for i in range(-3, 3)]),
                  ('vanderpol', 2, [0.1 * 2 ** i for i in range(0, 10)])]

    # pre-fill results with lists of  setups
    results = dict()
    for setup, nvars, param_list in setup_list:
        results[setup] = (nvars, param_list)

    # loop over all Q-delta matrix types
    for qd_type in qd_list:

        # assign implicit Q-delta matrix
        sweeper_params['QI'] = qd_type

        # loop over all setups
        for setup, nvars, param_list in setup_list:

            # initialize problem parameters (part I)
            problem_params = dict()
            problem_params['nvars'] = nvars  # number of degrees of freedom for each level

            # loop over all parameters
            for param in param_list:

                # fill description for the controller
                description = dict()
                description['dtype_u'] = mesh
                description['dtype_f'] = mesh
                description['sweeper_class'] = generic_implicit  # pass sweeper
                description['sweeper_params'] = sweeper_params  # pass sweeper parameters
                description['step_params'] = step_params  # pass step parameters

                print('working on: %s - %s - %s' % (qd_type, setup, param))

                # decide which setup to take
                if setup == 'heat':

                    problem_params['nu'] = param
                    problem_params['freq'] = 2

                    level_params['dt'] = 0.1

                    description['problem_class'] = heat1d
                    description['problem_params'] = problem_params
                    description['level_params'] = level_params  # pass level parameters

                elif setup == 'advection':

                    problem_params['c'] = param
                    problem_params['order'] = 2
                    problem_params['freq'] = 2

                    level_params['dt'] = 0.1

                    description['problem_class'] = advection1d
                    description['problem_params'] = problem_params
                    description['level_params'] = level_params  # pass level parameters

                elif setup == 'vanderpol':

                    problem_params['newton_tol'] = 1E-11
                    problem_params['maxiter'] = 50
                    problem_params['mu'] = param
                    problem_params['u0'] = np.array([2.0, 0])

                    level_params['dt'] = 0.1

                    description['problem_class'] = vanderpol
                    description['problem_params'] = problem_params
                    description['level_params'] = level_params

                else:
                    print('Setup not implemented..', setup)
                    exit()

                # instantiate controller
                controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params,
                                                         description=description)

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_exact(0)

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=0, Tend=0.1)

                # filter statistics by type (number of iterations)
                filtered_stats = filter_stats(stats, type='niter')

                # convert filtered statistics to list of iterations count, sorted by process
                iter_counts = sort_stats(filtered_stats, sortby='time')

                # just one time-step, grep number of iteration and store
                niter = iter_counts[0][1]
                id = ID(setup=setup, qd_type=qd_type, param=param)
                results[id] = niter

    # write out for later visualization, see preconditioner_plot.py
    file = open('results_iterations_precond.pkl', 'wb')
    pickle.dump(results, file)


if __name__ == "__main__":
    main()
