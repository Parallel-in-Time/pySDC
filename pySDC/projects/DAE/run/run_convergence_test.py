import numpy as np
import statistics
import pickle

# For notifications
import beepy

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier
from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats


def compute_convergence_data():
    """
    Routine to run convergence tests for the fully implicit solver using specified example with various preconditioners, time step sizes and collocation node counts
    Error data is stored in a dictionary and then pickled for use with the loglog_plot.py routine
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    
    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['newton_tol'] = 1e-6# tollerance for implicit solver
    problem_params['nvars'] = 5

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] =30

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = one_transistor_amplifier
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 1e-3
    # set simulation parameters
    # Note: one transistor model stable for three nodes for dt <= 1e-3
    num_samples = 10
    dt_list = np.logspace(-3, -5, num=num_samples)
    qd_list = ['LU']
    num_nodes_list = [3, 4, 5]
    conv_data = dict()  

    for qd_type in qd_list: 
        sweeper_params['QI'] = qd_type
        conv_data[qd_type] = dict()

        for num_nodes in num_nodes_list:
            sweeper_params['num_nodes'] = num_nodes 
            description['sweeper_params'] = sweeper_params
            conv_data[qd_type][num_nodes] = dict()
            conv_data[qd_type][num_nodes]['error'] = np.zeros_like(dt_list)
            conv_data[qd_type][num_nodes]['niter'] = np.zeros_like(dt_list, dtype='int')
            conv_data[qd_type][num_nodes]['dt'] = dt_list

            for j, dt in enumerate(dt_list):
                print('Working on Qdelta=%s -- num. nodes=%i -- dt=%f' % (qd_type, num_nodes, dt))
                level_params['dt'] = dt
                description['level_params'] = level_params

                # instantiate the controller
                controller = controller_nonMPI(
                    num_procs=1, controller_params=controller_params, description=description
                )
                # get initial values
                P = controller.MS[0].levels[0].prob
                uinit = P.u_exact(t0)

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                # compute exact solution and compare
                err = get_sorted(stats, type='error_post_step', sortby='time')
                niter = filter_stats(stats, type='niter')

                conv_data[qd_type][num_nodes]['error'][j] = np.linalg.norm([err[j][1] for j in range(len(err))], np.inf)
                conv_data[qd_type][num_nodes]['niter'][j] = round(statistics.mean(niter.values()))
                print("Error is", conv_data[qd_type][num_nodes]['error'][j])

    pickle.dump(conv_data, open("data/dae_conv_data.p", 'wb'))
    # beepy.beep(1)
    print("Done")


if __name__ == "__main__":
    compute_convergence_data()
