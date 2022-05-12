import numpy as np
import dill
from scipy.integrate import solve_ivp

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
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
                          sweep=L.status.sweep, type='u', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='p', value=L.uend[1])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)

def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2
    level_params['e_tol'] = 1e-7

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    problem_params = {           
        'mu'            :   5.,         
        'newton_tol'    :   1e-9,
        'newton_maxiter':   99,
        'u0'            :   np.array([2.0, 0.]),
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
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
    # uend, stats, uendall = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/piline.dat'
    fname = 'vdp.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


import matplotlib.pyplot as plt
import pickle
def my_plot(cwd='./'):
    fig, axs = plt.subplots(2,1, sharex=True)

    f = open(cwd + 'vdp.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    u = np.array(sort_stats(filter_stats(stats, type='u'), sortby='time'))[:,1]
    p = np.array(sort_stats(filter_stats(stats, type='p'), sortby='time'))[:,1]
    t = np.array(sort_stats(filter_stats(stats, type='p'), sortby='time'))[:,0]
    dt = np.array(sort_stats(filter_stats(stats, type='dt'), sortby='time'))[:,1]

    axs[0].plot(t, u)
    axs[0].plot(t, p)
    axs[1].plot(t, dt, label='pySDC')
    axs[1].set_yscale('log')

    try:
        with open('vdp_adaptivity_reference.pickle', 'rb') as f:
            ref = pickle.load(f)
         
        axs[1].plot(ref['t'][1:-1], ref['dt'][1:-1], linestyle='--', label='own code')
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
