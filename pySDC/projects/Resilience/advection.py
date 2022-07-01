import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD_periodic import advectionNd_periodic
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.error_estimation.accuracy_check import setup_mpl

from fault_injection import FaultInjector

num_procs = 1
np.random.seed = 0


class log_data(FaultInjector):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='u', value=L.uend)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_embedded', value=L.status.error_embedded_estimate)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_extrapolated', value=L.status.error_extrapolation_estimate)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='k', value=step.status.iter)


def run_advection(strategy, rng, faults, force_params=None):
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 0.0127
    level_params['e_tol'] = 1e-8

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'freq': (2,),
        'nvars': (2**7,),
        'c': 1.,
        'type': 'upwind',
        'order': 5,
        'ndim': 1,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['mssdc_jac'] = False
    controller_params['HotRod_tol'] = np.inf

    if strategy == 'adaptivity':
        controller_params['use_adaptivity'] = True
    elif strategy == 'HotRod':
        controller_params['use_HotRod'] = True
    elif strategy == 'iterate':
        step_params['maxiter'] = 100
        level_params['restol'] = 9.13e-10

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advectionNd_periodic  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # check if we want to change some parameters
    if force_params:
        for k in force_params.keys():
            for j in force_params[k].keys():
                if k == 'controller_params':
                    controller_params[j] = force_params[k][j]
                else:
                    description[k][j] = force_params[k][j]

    # set time parameters
    t0 = 0.0
    Tend = 2.5e-1
    Tend = 10 * level_params['dt']

    # instantiate controller
    controller_class = controller_nonMPI
    controller = controller_class(num_procs=num_procs, controller_params=controller_params,
                                  description=description)

    controller.hooks.random_generator = rng

    if faults:
        controller.hooks.add_random_fault(timestep=5, rnd_args={'iteration': 5})

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


def plot(stats, controller, strategy):
    setup_mpl()

    # convert filtered statistics to list of iterations count, sorted by process
    u = get_sorted(stats, type='u', sortby='time')
    t = np.array(get_sorted(stats, type='dt', sortby='time'))[:, 0]
    e_em = np.array(get_sorted(stats, type='e_embedded', sortby='time'))[:, 1]
    e_ex = np.array(get_sorted(stats, type='e_extrapolated', sortby='time'))[:, 1]
    ready = np.logical_and(e_ex != np.array(None), e_em != np.array(None))
    bitflips = get_sorted(stats, type='bitflip')

    # plot the solution
    sol_fig, sol_ax = plt.subplots(1, 1)
    x = controller.MS[0].levels[0].prob.xv[0]
    sol_ax.plot(x, u[0][1], label=r'$u_0$')
    sol_ax.plot(x, u[-1][1], label=r'$u_f$')

    sol_ax.set_xlabel(r'$x$')
    sol_ax.set_ylabel(r'$u$')
    sol_ax.legend(frameon=False)
    sol_fig.savefig(f'data/advection-sol-{strategy}.pdf', bbox_inches='tight')

    if strategy in ['HotRod']:
        # plot Hot Rod
        HR_fig, HR_ax = plt.subplots(1, 1, figsize=(3.5, 3))
        HR_ax.plot(t, e_em, label=r'$\epsilon_\mathrm{embedded}$')
        HR_ax.plot(t[e_ex != [None]], e_ex[e_ex != [None]], label=r'$\epsilon_\mathrm{extrapolated}$', ls='--',
                   marker='*')
        HR_ax.plot(t[ready], abs(e_em[ready] - e_ex[ready]), label=r'$\Delta$', ls='-.')

        # plot the faults
        for i in range(len(bitflips)):
            HR_ax.axvline(bitflips[i][0], color='grey', alpha=0.5)

        HR_ax.set_yscale('log')
        HR_ax.legend(frameon=False)
        HR_ax.set_yscale('log')

        HR_ax.set_xlabel('Time')
        HR_ax.set_ylabel(r'$\Delta t$')

        HR_fig.savefig(f'data/advection_hotrod_{strategy}.png', bbox_inches='tight', dpi=300)


def main():
    for strategy in ['adaptivity', 'nothing', 'iterate']:
        stats, controller, Tend = run_advection(strategy, None, False)
        plot(stats, controller, strategy)


if __name__ == "__main__":
    main()
