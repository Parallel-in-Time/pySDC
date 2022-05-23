import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.error_estimation.accuracy_check import setup_mpl

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
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_embedded', value=L.status.error_embedded_estimate)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_extrapolated', value=L.status.error_extrapolation_estimate)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restart', value=1, initialize=0)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='sweeps', value=step.status.iter)


def run(use_adaptivity=True):
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 3e-2
    level_params['e_tol'] = 1e-8

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'Vs': 100.,
        'Rs': 1.,
        'C1': 1.,
        'Rpi': 0.2,
        'C2': 1.,
        'Lpi': 1.,
        'Rl': 5.,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['use_HotRod'] = True
    controller_params['use_adaptivity'] = use_adaptivity

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
    controller_class = controller_nonMPI
    controller = controller_class(num_procs=1, controller_params=controller_params,
                                  description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats


def plot(stats, use_adaptivity):
    setup_mpl()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = np.array(sort_stats(filter_stats(stats, type='v1'), sortby='time'))[:, 1]
    v2 = np.array(sort_stats(filter_stats(stats, type='v2'), sortby='time'))[:, 1]
    p3 = np.array(sort_stats(filter_stats(stats, type='p3'), sortby='time'))[:, 1]
    t = np.array(sort_stats(filter_stats(stats, type='p3'), sortby='time'))[:, 0]
    dt = np.array(sort_stats(filter_stats(stats, type='dt'), sortby='time'))[:, 1]
    e_em = np.array(sort_stats(filter_stats(stats, type='e_embedded'), sortby='time'))[:, 1]
    e_ex = np.array(sort_stats(filter_stats(stats, type='e_extrapolated'), sortby='time'))[:, 1]
    restarts = np.array(sort_stats(filter_stats(stats, type='restart'), sortby='time'))[:, 1]
    sweeps = np.array(sort_stats(filter_stats(stats, type='sweeps'), sortby='time'))[:, 1]
    ready = np.logical_and(e_ex != np.array(None), e_em != np.array(None))

    assert np.allclose([v1[-1], v2[-2], p3[-1]], [83.88431516810506, 80.62596592922169, 16.134334413301595], rtol=1),\
        'Solution is wrong!'

    if use_adaptivity:
        assert np.isclose(e_em[-1], 6.48866205210652e-09), f'Embedded error estimate when using adaptivity is wrong!\
Expected {6.5e-9:.1e}, got {e_em[-1]:.1e}'
        assert np.isclose(e_ex[-1], 6.565837120935149e-09), f'Extrapolation error when using adaptivity estimate is wro\
ng! Expected {6.6e-9:.1e}, got {e_ex[-1]:.1e}.'
        assert np.isclose(dt[-1], 0.0535436291079129), 'Time step size is wrong!'
        assert np.isclose(restarts.sum(), 1), f'Expected 1 restart, got {restarts.sum()}'
        assert np.isclose(sweeps.sum(), 4296), f'Expected 4296 sweeps, got {sweeps.sum()}'
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        ax.plot(t, v1, label='v1', ls='-')
        ax.plot(t, v2, label='v2', ls='--')
        ax.plot(t, p3, label='p3', ls='-.')
        ax.legend(frameon=False)
        ax.set_xlabel('Time')
        fig.savefig('data/piline_solution_adaptive.png', bbox_inches='tight', dpi=300)
    else:
        assert np.isclose(e_em[-1], 6.510703087769798e-10), 'Embedded error estimate is wrong!'
        assert np.isclose(e_ex[-1], 6.541069907939809e-10), 'Extrapolation error estimate is wrong!'
        assert np.isclose(dt[-1], 3e-2), 'Time step size is wrong!'
        assert np.isclose(restarts.sum(), 0), f'Expected 0 restarts, got {restarts.sum()}'
        assert np.isclose(sweeps.sum(), 2668), f'Expected 2668 sweeps, got {sweeps.sum()}'

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    ax.plot(t, dt, color='black')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    e_ax = ax.twinx()
    e_ax.plot(t, e_em, label=r'$\epsilon_\mathrm{embedded}$')
    e_ax.plot(t, e_ex, label=r'$\epsilon_\mathrm{extrapolated}$', ls='--')
    e_ax.plot(t[ready], abs(e_em[ready] - e_ex[ready]), label='difference', ls='-.')
    e_ax.plot([None, None], label=r'$\Delta t$', color='black')
    e_ax.set_yscale('log')
    if use_adaptivity:
        e_ax.legend(frameon=False, loc='upper left')
    else:
        e_ax.legend(frameon=False, loc='upper right')
    e_ax.set_ylim((1e-13, 5e-6))
    ax.set_ylim((8e-3, 58e-3))

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\Delta t$')

    if use_adaptivity:
        fig.savefig('data/piline_hotrod_adaptive.png', bbox_inches='tight', dpi=300)
    else:
        fig.savefig('data/piline_hotrod.png', bbox_inches='tight', dpi=300)


def main():
    for use_adaptivity in [False, True]:
        stats = run(use_adaptivity)
        plot(stats, use_adaptivity)


if __name__ == "__main__":
    main()
