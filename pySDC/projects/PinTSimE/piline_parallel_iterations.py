import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import dill

from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.PinTSimE.piline_model import log_data, setup_mpl

import pySDC.helpers.plot_helper as plt_helper

def run_test_parallel(dt):
    """
        A simple test program to run PFASST and compute residuals for different time steps
    """
    
    # set MPI communicator
    comm = MPI.COMM_WORLD

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = dt

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

    assert 'errtol' not in description['step_params'].keys(), "No exact or reference solution known to compute error"

    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

    # set time parameters
    t0 = 0.0
    Tend = 15

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # combine statistics into list of statistics
    iter_counts_list = comm.gather(iter_counts, root=0)
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
    
        # build one list of statistics instead of list of lists, the sort by time
        iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
        
        niters = np.array([item[1] for item in iter_counts])
        iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])
