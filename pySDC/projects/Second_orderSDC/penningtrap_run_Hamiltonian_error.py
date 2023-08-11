import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN

from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param

def get_ham_error(tend, dt, maxiter, sweeper):
    controller_params, description= penningtrap_param()
    description['level_params']['dt']=dt
    description['sweeper_params']['num_nodes']=3
    description['problem_params']['omega_E']=1
    description['problem_params']['omega_B']=0
    description['problem_params']['u0']=np.array([[0, 0, 0], [0, 0, 1], [1], [1]], dtype=object)
    description['sweeper_class']=sweeper
    description['step_params']['maxiter']=maxiter

    controller=controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    penningtrap.Harmonic_oscillator=True
    t0=0.0
    Tend=tend

    P = controller.MS[0].levels[0].prob
    uinit=P.u_init()

    uend, stats= controller.run(u0=uinit, t0=t0, Tend=Tend)

    sortedlist_stats=get_sorted(stats, type='etot', sortby='time')

    H0=0.5*(np.dot(uinit.vel[:].T, uinit.vel[:]) + np.dot(uinit.pos[:].T, uinit.pos[:]))

    Hamiltonian_err = np.ravel([abs(entry[1] - H0) / H0 for entry in sortedlist_stats])

    return Hamiltonian_err

def plot_hamiltonian(Hamiltonian_error, Tend, dt, maxiter_list, step=3000):
    time=np.arange(0, Tend, dt)
    t_len=len(time)
    plt.figure()
    plt.loglog(time[:t_len:step], Hamiltonian_error['RKN'][:t_len:step], marker='.', ls=' ', label='RKN')
    plt.loglog(time[:t_len:step], Hamiltonian_error['SDC2'][:t_len:step], marker='s', ls=' ', label=f'k={maxiter_list[0]}')
    plt.loglog(time[:t_len:step], Hamiltonian_error['SDC3'][:t_len:step], marker='*', ls=' ', label=f'k={maxiter_list[1]}')
    plt.loglog(time[:t_len:step], Hamiltonian_error['SDC4'][:t_len:step], marker='H', ls=' ', label=f'k={maxiter_list[2]}')
    plt.ylabel('$\Delta H^{\mathrm{(rel)}}$')
    plt.xlabel('$\omega \cdot t$')
    plt.legend()
    plt.tight_layout()

if __name__=='__main__':
    maxiter_list=(2, 3, 4)
    tend=2*1e+6
    dt=2*np.pi/10
    Hamiltonian_error=dict()
    Hamiltonian_error['SDC2']=get_ham_error(tend, dt, maxiter_list[0], boris_2nd_order)
    # np.save('Ham_SDC2.npy', Ham_SDC2)

    Hamiltonian_error['SDC3']=get_ham_error(tend, dt, maxiter_list[1], boris_2nd_order)
    # np.save('Ham_SDC3.npy', Ham_SDC3)

    Hamiltonian_error['SDC4']=get_ham_error(tend, dt, maxiter_list[2], boris_2nd_order)
    # np.save('Ham_SDC4.npy', Ham_SDC4)

    Hamiltonian_error['RKN']=get_ham_error(tend, dt, 1, RKN)
    # np.save('Ham_RKN.npy', Ham_RKN)

    plot_hamiltonian(Hamiltonian_error, tend, dt, maxiter_list)
