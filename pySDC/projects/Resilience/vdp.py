# script to run a van der Pol problem
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.core.Errors import ProblemError


def plot_step_sizes(stats, ax):

    # convert filtered statistics to list of iterations count, sorted by process
    u = np.array(get_sorted(stats, type='u', recomputed=False, sortby='time'))[:, 1]
    p = np.array(get_sorted(stats, type='p', recomputed=False, sortby='time'))[:, 1]
    t = np.array(get_sorted(stats, type='p', recomputed=False, sortby='time'))[:, 0]

    e_em = np.array(get_sorted(stats, type='e_em', recomputed=False, sortby='time'))[:, 1]
    dt = np.array(get_sorted(stats, type='dt', recomputed=False, sortby='time'))
    restart = np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))

    ax.plot(t, u, label=r'$u$')
    ax.plot(t, p, label=r'$p$')

    dt_ax = ax.twinx()
    dt_ax.plot(dt[:, 0], dt[:, 1], color='black')
    dt_ax.plot(t, e_em, color='magenta')
    dt_ax.set_yscale('log')
    dt_ax.set_ylim((5e-10, 3e-1))

    ax.plot([None], [None], label=r'$\Delta t$', color='black')
    ax.plot([None], [None], label=r'$\epsilon_\mathrm{embedded}$', color='magenta')
    ax.plot([None], [None], label='restart', color='grey', ls='-.')

    for i in range(len(restart)):
        if restart[i, 1] > 0:
            ax.axvline(restart[i, 0], color='grey', ls='-.')
    ax.legend(frameon=False)

    ax.set_xlabel('time')


class log_data(hooks):
    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='p',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='dt',
            value=L.dt,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_em',
            value=L.status.error_embedded_estimate,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_ex',
            value=L.status.error_extrapolation_estimate,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.restart),
        )


def run_vdp(
    custom_description=None,
    num_procs=1,
    Tend=10.0,
    hook_class=log_data,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
):

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        controller.hooks.random_generator = fault_stuff['rng']
        controller.hooks.add_fault(
            rnd_args={'iteration': 3, **fault_stuff.get('rnd_params', {})},
            args={'time': 1.0, 'target': 0, **fault_stuff.get('args', {})},
        )

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ProblemError:
        stats = controller.hooks.return_stats()

    return stats, controller, Tend
