import numpy as np
import dill

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.playgrounds.EnergyGrids.log_data_battery import log_data_battery
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to do PFASST runs for the battery drain model
    """
    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 1e-3
    
    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 5
    problem_params['V_ref'] = 1
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data_battery
    
    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    
    # set time parameters
    t0 = 0.0
    Tend = 3
    
    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    
    # fname = 'data/piline.dat'
    fname = 'battery.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()
    
def plot_voltages(cwd='./'):
    f = open(cwd + 'battery.dat', 'rb')
    stats = dill.load(f)
    f.close()
    
    # convert filtered statistics to list of iterations count, sorted by process
    cL = sort_stats(filter_stats(stats, type='current L'), sortby='time')
    vC = sort_stats(filter_stats(stats, type='voltage C'), sortby='time')
    
    times = [v[0] for v in cL]

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(times, [v[1] for v in cL], label='current L')
    plt_helper.plt.plot(times, [v[1] for v in vC], label='voltage C')
    plt_helper.plt.legend()
    
    plt_helper.plt.show()


if __name__ == "__main__":
    main()
    plot_voltages()
