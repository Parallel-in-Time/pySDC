import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import dill
import time

from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.PinTSimE.piline_model import log_data, setup_mpl

import pySDC.helpers.plot_helper as plt_helper


def set_parameters_ml():
    """
        Helper routine to set parameters for the following multi-level runs

        Returns:
            dict: dictionary containing the simulation parameters
            dict: dictionary containing the controller parameters
            float: starting time
            float: end time
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = 1E-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
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
    
    # set time parameters
    t0 = 0.0
    Tend = 15
    
    return description, controller_params, t0, Tend


def main():
    """
        Test program to run the Pi-line model in parallel using PFASST
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    description, controller_params, t0, Tend = set_parameters_ml()

    assert 'errtol' not in description['step_params'].keys(), "No exact or reference solution known to compute error"

    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # combine statistics into list of statistics
    iter_counts_list = comm.gather(iter_counts, root=0)

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = get_sorted(stats, type='v1', sortby='time')
    v2 = get_sorted(stats, type='v2', sortby='time')
    p3 = get_sorted(stats, type='p3', sortby='time')

    v1_list = comm.gather(v1, root=0)
    v2_list = comm.gather(v2, root=0)
    p3_list = comm.gather(p3, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        # compute and print statistics
        min_iter = 20
        max_iter = 0

        f = open('piline_parallel_out.txt', 'w')
        out = 'Working with %2i processes...' % size
        f.write(out + '\n')
        print(out)

        # build one list of statistics instead of list of lists, the sort by time
        iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
        iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])

        niters = np.array([item[1] for item in iter_counts])
        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        f.write(out + '\n')
        print(out)

        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %1i' % item
            f.write(out + '\n')
            print(out)
            min_iter = min(min_iter, item[1])
            max_iter = max(max_iter, item[1])

        assert np.mean(niters) <= 10, "Mean number of iterations is too high, got %s" % np.mean(niters)
        f.close()
    
        #plot_voltages(v1_list, v2_list, p3_list)
        count_time()

def plot_voltages(v1_list, v2_list, p3_list, cwd='./'):

    # build one list of statistics instead of list of lists, the sort by time
    v1_gather = [item for sublist in v1_list for item in sublist]
    v1 = sorted(v1_gather, key=lambda tup: tup[0])
    v2_gather = [item for sublist in v2_list for item in sublist]
    v2 = sorted(v2_gather, key=lambda tup: tup[0])
    p3_gather = [item for sublist in p3_list for item in sublist]
    p3 = sorted(p3_gather, key=lambda tup: tup[0])

    times = [v[0] for v in v1]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in v1], linewidth=1, label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in v2], linewidth=1, label='$v_{C_2}$')
    ax.plot(times, [v[1] for v in p3], linewidth=1, label='$i_{L_\pi}$')
    ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('piline_model_parallel_solution.png', dpi=300, bbox_inches='tight')
    
def count_time():
    """
        Routine to run PFASST with different numbers of 
    """
    
    # set MPI communicator
    comm = MPI.COMM_WORLD
    
    description, controller_params, t0, Tend = set_parameters_ml()
    
    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
    
    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)
    
    start_time = time.perf_counter()
    
    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    
    end_time = time.perf_counter()
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    count_times_rank = end_time - start_time
    
    count_times_list = comm.gather(count_times_rank, root=0)
    
    print(count_times_list)


if __name__ == "__main__":
    main()
