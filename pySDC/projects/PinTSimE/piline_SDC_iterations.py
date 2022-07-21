import matplotlib as mpl
import numpy as np
import dill

mpl.use('Agg')

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.PinTSimE.piline_model import log_data, setup_mpl
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
import pySDC.helpers.plot_helper as plt_helper


def run_test(dt, num_procs, M=None):
    """
        A simple test program to run SDC/PFASST (nonMPI) and compute residuals for different numbers of GL-nodes
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    if M is not None:
        sweeper_params['num_nodes'] = M

    else:
        sweeper_params['num_nodes'] = [3, 5]
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 100.0
    problem_params['Rs'] = 1.0
    problem_params['C1'] = 1.0
    problem_params['Rpi'] = 0.2
    problem_params['C2'] = 1.0
    problem_params['Lpi'] = 1.0
    problem_params['Rl'] = 5.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline                         # pass problem class
    description['problem_params'] = problem_params                # pass problem parameters
    description['sweeper_class'] = imex_1st_order                 # pass sweeper
    description['sweeper_params'] = sweeper_params                # pass sweeper parameters
    description['level_params'] = level_params                    # pass level parameters
    description['step_params'] = step_params                      # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh            # pass spatial transfer class

    assert 'errtol' not in description['step_params'].keys(), "No exact or reference solution known to compute error"

    # set time parameters
    t0 = 0.0
    Tend = 15

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats

def plot_SDC_iterations():
    """
        Routine to plot the residuals for each SDC run
    """

    num_nodes = [3, 5, 7]
    dt_list = [1e-2, 1e-3, 1e-4, 1e-5]
    num_procs = 1
    iter_nodes_asarray = np.zeros((len(dt_list), len(num_nodes)))

    i, j = 0, 0
    for dt in dt_list:
        for M in num_nodes:

            stats = run_test(dt, num_procs, M)

            iter_counts = get_sorted(stats, type='niter', sortby='time')

            niters = np.array([item[1] for item in iter_counts])

            iter_nodes_asarray[i, j] = np.mean(niters) 

            j += 1

        j = 0
        i += 1

    setup_mpl()
    color = ['green', 'blue', 'red']
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    for j in range(len(num_nodes)):
        ax.plot(dt_list, iter_nodes_asarray[:, j], color = color[j], marker = 's',
                label='M={}'.format(num_nodes[j]))

    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.set_xlabel('$\Delta t$')
    ax.set_ylabel('Average number of iterations')
    ax.set_xscale('log', base=10)

    fig.savefig('piline_SDC_iterations.png', dpi=300, bbox_inches='tight')
    
def plot_PFASST_iterations():
    """
        Routine to plot the residuals for each PFASST run
    """

    dt_list = [1e-2, 1e-3]
    num_procs = 8
    iter_nodes_asarray = np.zeros((len(dt_list)))

    i = 0
    for dt in dt_list:

        stats = run_test(dt, num_procs)

        iter_counts = get_sorted(stats, type='niter', sortby='time')

        niters = np.array([item[1] for item in iter_counts])

        iter_nodes_asarray[i] = np.mean(niters) 

        i += 1

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.plot(dt_list, iter_nodes_asarray, color = 'blue', marker = 's',
                label='M=[3,5]')

    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.set_xlabel('$\Delta t$')
    ax.set_ylabel('Average number of iterations')
    ax.set_xscale('log', base=10)

    fig.savefig('piline_PFASST_iterations.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    plot_PFASST_iterations()
