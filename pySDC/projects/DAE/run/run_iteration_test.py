import numpy as np
import statistics
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats


def setup():
    """
    Routine to run simple differential-algebraic-equation example with various max iters, preconditioners and collocation node counts
    In contrast to run_convergence_test.py, in which max iters is set large enough to not be the limiting factor, max iters is varied for a fixed time step and the improvement in the error is measured
    Error data is stored in a dictionary and then pickled for use with the loglog_plot.py routine
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = 1e-3

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    
    # This comes as read-in for the problem class
    problem_params = dict()
    # Absolute termination tollerance for implicit solver
    # Exactly how this is used can be adjusted in update_nodes() in the fully implicit sweeper
    problem_params['newton_tol'] = 1e-7
    problem_params['nvars'] = 3 

    # This comes as read-in for the step class
    step_params = dict()

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params

    # set simulation parameters
    run_params = dict()
    run_params['t0'] = 0.0
    run_params['tend'] = 0.1
    max_iter_low = 4
    max_iter_high = 10
    run_params['max_iter_list'] = [i for i in range(max_iter_low, max_iter_high)]
    run_params['qd_list'] = ['IE', 'LU']
    run_params['num_nodes_list'] = [3]

    return description, controller_params, run_params


def run(description, controller_params, run_params):

    conv_data = dict()

    for qd_type in run_params['qd_list']: 
        description['sweeper_params']['QI'] = qd_type
        conv_data[qd_type] = dict()

        for num_nodes in run_params['num_nodes_list']:
            description['sweeper_params']['num_nodes'] = num_nodes 
            conv_data[qd_type][num_nodes] = dict()
            conv_data[qd_type][num_nodes]['error'] = np.zeros_like(run_params['max_iter_list'], dtype=float)
            conv_data[qd_type][num_nodes]['residual'] = np.zeros_like(run_params['max_iter_list'], dtype=float)
            conv_data[qd_type][num_nodes]['niter'] = np.zeros_like(run_params['max_iter_list'], dtype='int')
            conv_data[qd_type][num_nodes]['max_iter'] = run_params['max_iter_list']

            for i, max_iter in enumerate(run_params['max_iter_list']):
                print('Working on Qdelta=%s -- num. nodes=%i -- max. iter.=%i' % (qd_type, num_nodes, max_iter))
                description['step_params'] = dict()
                description['step_params']['maxiter'] = max_iter

                # instantiate the controller
                controller = controller_nonMPI(
                    num_procs=1, controller_params=controller_params, description=description
                )
                # get initial values
                P = controller.MS[0].levels[0].prob
                uinit = P.u_exact(run_params['t0'])

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=run_params['t0'], Tend=run_params['tend'])

                # compute exact solution and compare
                err = get_sorted(stats, type='error_post_step', sortby='time')
                residual = get_sorted(stats, type='residual_post_step', sortby='time')
                niter = filter_stats(stats, type='niter')

                conv_data[qd_type][num_nodes]['error'][i] = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
                conv_data[qd_type][num_nodes]['residual'][i] = np.linalg.norm([residual[i][1] for i in range(len(residual))], np.inf)
                conv_data[qd_type][num_nodes]['niter'][i] = round(statistics.mean(niter.values()))
                print("Error=", conv_data[qd_type][num_nodes]['error'][i], ".  Residual=", conv_data[qd_type][num_nodes]['residual'][i])
                
    return conv_data
    


def main(): 
    description, controller_params, run_params = setup()
    conv_data = run(description, controller_params, run_params)
    pickle.dump(conv_data, open("data/dae_conv_data.p", 'wb'))
    print("Done")


if __name__ == "__main__":
    main()
