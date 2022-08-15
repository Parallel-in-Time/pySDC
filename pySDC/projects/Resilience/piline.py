import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.hotrod import HotRod

from pySDC.core.Hooks import hooks


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
            type='v1',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='v2',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='p3',
            value=L.uend[2],
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
            type='e_embedded',
            value=L.status.error_embedded_estimate,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_extrapolated',
            value=L.status.error_extrapolation_estimate,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=1,
            initialize=0,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='sweeps',
            value=step.status.iter,
        )


def run_piline(custom_description, num_procs, Tend=20.0, hook_class=log_data):
    """
    A simple test program to do SDC runs for Piline problem
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'Vs': 100.0,
        'Rs': 1.0,
        'C1': 1.0,
        'Rpi': 0.2,
        'C2': 1.0,
        'Lpi': 1.0,
        'Rl': 5.0,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    description.update(custom_description)

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller_class = controller_nonMPI
    controller = controller_class(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats


def get_data(stats, recomputed=False):
    # convert filtered statistics to list of iterations count, sorted by process
    data = {
        'v1': np.array(get_sorted(stats, type='v1', recomputed=recomputed))[:, 1],
        'v2': np.array(get_sorted(stats, type='v2', recomputed=recomputed))[:, 1],
        'p3': np.array(get_sorted(stats, type='p3', recomputed=recomputed))[:, 1],
        't': np.array(get_sorted(stats, type='p3', recomputed=recomputed))[:, 0],
        'dt': np.array(get_sorted(stats, type='dt', recomputed=recomputed)),
        'e_em': np.array(get_sorted(stats, type='e_embedded', recomputed=recomputed))[:, 1],
        'e_ex': np.array(get_sorted(stats, type='e_extrapolated', recomputed=recomputed))[:, 1],
        'restarts': np.array(get_sorted(stats, type='restart', recomputed=recomputed))[:, 1],
        'sweeps': np.array(get_sorted(stats, type='sweeps', recomputed=recomputed))[:, 1],
    }
    data['ready'] = np.logical_and(data['e_ex'] != np.array(None), data['e_em'] != np.array(None))
    return data


def plot_error(data, ax, use_adaptivity=True):
    setup_mpl_from_accuracy_check()
    ax.plot(data['dt'][:, 0], data['dt'][:, 1], color='black')

    e_ax = ax.twinx()
    e_ax.plot(data['t'], data['e_em'], label=r'$\epsilon_\mathrm{embedded}$')
    e_ax.plot(data['t'][data['ready']], data['e_ex'][data['ready']], label=r'$\epsilon_\mathrm{extrapolated}$', ls='--')
    e_ax.plot(
        data['t'][data['ready']],
        abs(data['e_em'][data['ready']] - data['e_ex'][data['ready']]),
        label='difference',
        ls='-.',
    )

    e_ax.plot([None, None], label=r'$\Delta t$', color='black')
    e_ax.set_yscale('log')
    if use_adaptivity:
        e_ax.legend(frameon=False, loc='upper left')
    else:
        e_ax.legend(frameon=False, loc='upper right')
    e_ax.set_ylim((7.367539795147197e-12, 1.109667868425781e-05))
    ax.set_ylim((0.012574322653781072, 0.10050387672423527))

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\Delta t$')
    ax.set_xlabel('Time')


def setup_mpl_from_accuracy_check():
    from pySDC.projects.Resilience.accuracy_check import setup_mpl

    setup_mpl()


def plot_solution(data, ax):
    setup_mpl_from_accuracy_check()
    ax.plot(data['t'], data['v1'], label='v1', ls='-')
    ax.plot(data['t'], data['v2'], label='v2', ls='--')
    ax.plot(data['t'], data['p3'], label='p3', ls='-.')
    ax.legend(frameon=False)
    ax.set_xlabel('Time')


def check_solution(data, use_adaptivity, num_procs, generate_reference=False):
    if use_adaptivity and num_procs == 1:
        error_msg = 'Error when using adaptivity in serial:'
        expected = {
            'v1': 83.88330442715265,
            'v2': 80.62692930055763,
            'p3': 16.13594155613822,
            'e_em': 4.922608098922865e-09,
            'e_ex': 4.4120077421613226e-08,
            'dt': 0.05,
            'restarts': 1.0,
            'sweeps': 2416.0,
            't': 20.03656747407325,
        }

    elif use_adaptivity and num_procs == 4:
        error_msg = 'Error when using adaptivity in parallel:'
        expected = {
            'v1': 83.88400082289273,
            'v2': 80.62656229801286,
            'p3': 16.134850400599763,
            'e_em': 2.3681899108396465e-08,
            'e_ex': 3.6491178375304526e-08,
            'dt': 0.08265581329617167,
            'restarts': 8.0,
            'sweeps': 2432.0,
            't': 19.999999999999996,
        }

    elif not use_adaptivity and num_procs == 4:
        error_msg = 'Error with fixed step size in parallel:'
        expected = {
            'v1': 83.88400128006428,
            'v2': 80.62656202423844,
            'p3': 16.134849781053525,
            'e_em': 4.277040943634347e-09,
            'e_ex': 4.9707053288253756e-09,
            'dt': 0.05,
            'restarts': 0.0,
            'sweeps': 1600.0,
            't': 20.00000000000015,
        }

    elif not use_adaptivity and num_procs == 1:
        error_msg = 'Error with fixed step size in serial:'
        expected = {
            'v1': 83.88400149770143,
            'v2': 80.62656173487008,
            'p3': 16.134849851184736,
            'e_em': 4.977994905175365e-09,
            'e_ex': 5.048084913047097e-09,
            'dt': 0.05,
            'restarts': 0.0,
            'sweeps': 1600.0,
            't': 20.00000000000015,
        }

    got = {
        'v1': data['v1'][-1],
        'v2': data['v2'][-1],
        'p3': data['p3'][-1],
        'e_em': data['e_em'][-1],
        'e_ex': data['e_ex'][data['e_ex'] != [None]][-1],
        'dt': data['dt'][-1][1],
        'restarts': data['restarts'].sum(),
        'sweeps': data['sweeps'].sum(),
        't': data['t'][-1],
    }

    if generate_reference:
        print(f'Adaptivity: {use_adaptivity}, num_procs={num_procs}')
        print('expected = {')
        for k in got.keys():
            v = got[k]
            if type(v) in [list, np.ndarray]:
                print(f'    \'{k}\': {v[v!=[None]][-1]},')
            else:
                print(f'    \'{k}\': {v},')
        print('}')

    for k in expected.keys():
        assert np.isclose(
            expected[k], got[k], rtol=1e-4
        ), f'{error_msg} Expected {k}={expected[k]:.2e}, got {k}={got[k]:.2e}'


def main():
    generate_reference = False

    for use_adaptivity in [True, False]:
        custom_description = {'convergence_controllers': {}}
        if use_adaptivity:
            custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7}

        for num_procs in [1, 4]:
            custom_description['convergence_controllers'][HotRod] = {'HotRod_tol': 1, 'no_storage': num_procs > 1}
            stats = run_piline(custom_description, num_procs=num_procs)
            data = get_data(stats)
            check_solution(data, use_adaptivity, num_procs, generate_reference)
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
            plot_error(data, ax, use_adaptivity)
            if use_adaptivity:
                fig.savefig(f'data/piline_hotrod_adaptive_{num_procs}procs.png', bbox_inches='tight', dpi=300)
            else:
                fig.savefig(f'data/piline_hotrod_{num_procs}procs.png', bbox_inches='tight', dpi=300)
            if use_adaptivity and num_procs == 4:
                sol_fig, sol_ax = plt.subplots(1, 1, figsize=(3.5, 3))
                plot_solution(data, sol_ax)
                sol_fig.savefig('data/piline_solution_adaptive.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
