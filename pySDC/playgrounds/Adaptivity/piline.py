import numpy as np
import dill
from scipy.integrate import solve_ivp

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.BuckConverter import buck_converter
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.helpers.plot_helper as plt_helper

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from mpi4py import MPI
from controller_nonMPI_adaptive import controller_nonMPI_adaptive


from pySDC.core.Hooks import hooks


class log_data(hooks):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v1', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v2', value=L.uend[1])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='p3', value=L.uend[2])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)

def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2
    level_params['e_tol'] = 1e-8

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {           
        'Vs'    :   100.,
        'Rs'    :   1.,  
        'C1'    :   1.,  
        'Rpi'   :   0.2, 
        'C2'    :   1.,  
        'Lpi'   :   1.,  
        'Rl'    :   5.,  
        'c':1.,          
        'f':4.,          
        'mu':1.,         
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 2e1

    # instantiate controller
    controller_class = controller_nonMPI_adaptive
    controller = controller_class(num_procs=1, controller_params=controller_params,
                                   description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'piline.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


import matplotlib.pyplot as plt
import pickle
def my_plot(cwd='./'):
    fig, axs = plt.subplots(2,1, sharex=True)

    f = open(cwd + 'piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = np.array(sort_stats(filter_stats(stats, type='v1'), sortby='time'))[:,1]
    v2 = np.array(sort_stats(filter_stats(stats, type='v2'), sortby='time'))[:,1]
    p3 = np.array(sort_stats(filter_stats(stats, type='p3'), sortby='time'))[:,1]
    t = np.array(sort_stats(filter_stats(stats, type='p3'), sortby='time'))[:,0]
    dt = np.array(sort_stats(filter_stats(stats, type='dt'), sortby='time'))[:,1]


    axs[0].plot(t, v1)
    axs[0].plot(t, v2)
    axs[0].plot(t, p3)
    axs[1].plot(t, dt, label='pySDC')

    try:
        with open('piline_reference.pickle', 'rb') as f:
            ref = pickle.load(f)
         
        axs[1].plot(ref['t'][0:-2], ref['dt'][1:-1], linestyle='--', label='own code')
    except:
        pass

    axs[1].legend(frameon=False)

    axs[1].set_xlabel('time')
    axs[1].set_ylabel(r'$\Delta t$')
     
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    my_plot()
