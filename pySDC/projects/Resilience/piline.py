import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
from pySDC.projects.Resilience.hook import LogData, hook_collection
from pySDC.projects.Resilience.strategies import merge_descriptions


def run_piline(
    custom_description=None,
    num_procs=1,
    Tend=20.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
):
    """
    Run a Piline problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
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
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        rnd_args = {'iteration': 4}
        args = {'time': 2.5, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


def get_data(stats, recomputed=False):
    """
    Extract useful data from the stats.

    Args:
        stats (pySDC.stats): The stats object of the run
        recomputed (bool): Whether to exclude values that don't contribute to the final solution or not

    Returns:
        dict: Data
    """
    data = {
        'v1': np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=recomputed)]),
        'v2': np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=recomputed)]),
        'p3': np.array([me[1][2] for me in get_sorted(stats, type='u', recomputed=recomputed)]),
        't': np.array([me[0] for me in get_sorted(stats, type='u', recomputed=recomputed)]),
        'dt': np.array([me[1] for me in get_sorted(stats, type='dt', recomputed=recomputed)]),
        't_dt': np.array([me[0] for me in get_sorted(stats, type='dt', recomputed=recomputed)]),
        'e_em': np.array(get_sorted(stats, type='error_embedded_estimate', recomputed=recomputed))[:, 1],
        'e_ex': np.array(get_sorted(stats, type='error_extrapolation_estimate', recomputed=recomputed))[:, 1],
        'restarts': np.array(get_sorted(stats, type='restart', recomputed=None))[:, 1],
        't_restarts': np.array(get_sorted(stats, type='restart', recomputed=None))[:, 0],
        'sweeps': np.array(get_sorted(stats, type='sweeps', recomputed=None))[:, 1],
    }
    data['ready'] = np.logical_and(data['e_ex'] != np.array(None), data['e_em'] != np.array(None))
    data['restart_times'] = data['t_restarts'][data['restarts'] > 0]
    return data


def plot_error(data, ax, use_adaptivity=True, plot_restarts=False):
    """
    Plot the embedded and extrapolated error estimates.

    Args:
        data (dict): Data prepared from stats by `get_data`
        use_adaptivity (bool): Whether adaptivity was used
        plot_restarts (bool): Whether to plot vertical lines for restarts

    Returns:
        None
    """
    setup_mpl_from_accuracy_check()
    ax.plot(data['t_dt'], data['dt'], color='black')

    e_ax = ax.twinx()
    e_ax.plot(data['t'], data['e_em'], label=r'$\epsilon_\mathrm{embedded}$')
    e_ax.plot(data['t'][data['ready']], data['e_ex'][data['ready']], label=r'$\epsilon_\mathrm{extrapolated}$', ls='--')
    e_ax.plot(
        data['t'][data['ready']],
        abs(data['e_em'][data['ready']] - data['e_ex'][data['ready']]),
        label='difference',
        ls='-.',
    )

    if plot_restarts:
        [ax.axvline(t_restart, ls='-.', color='black', alpha=0.5) for t_restart in data['restart_times']]

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
    """
    Change matplotlib parameters to conform to LaTeX style.
    """
    from pySDC.projects.Resilience.accuracy_check import setup_mpl

    setup_mpl()


def plot_solution(data, ax):
    """
    Plot the solution.

    Args:
        data (dict): Data prepared from stats by `get_data`
        ax: Somewhere to plot

    Returns:
        None
    """
    setup_mpl_from_accuracy_check()
    ax.plot(data['t'], data['v1'], label='v1', ls='-')
    ax.plot(data['t'], data['v2'], label='v2', ls='--')
    ax.plot(data['t'], data['p3'], label='p3', ls='-.')
    ax.legend(frameon=False)
    ax.set_xlabel('Time')


def check_solution(data, use_adaptivity, num_procs, generate_reference=False):
    """
    Check the solution against a hard coded reference.

    Args:
        data (dict): Data prepared from stats by `get_data`
        use_adaptivity (bool): Whether adaptivity was used
        num_procs (int): Number of steps for MSSDC
        generate_reference (bool): Instead of comparing to reference, print a new reference to the console

    Returns:
        None
    """
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
            'v1': 83.88320903115796,
            'v2': 80.6269822629629,
            'p3': 16.136084724243805,
            'e_em': 4.0668446388281154e-09,
            'e_ex': 4.901094641240463e-09,
            'dt': 0.05,
            'restarts': 48.0,
            'sweeps': 2592.0,
            't': 20.041499821475185,
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
        'dt': data['dt'][-1],
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
        ), f'{error_msg} Expected {k}={expected[k]:.4e}, got {k}={got[k]:.4e}'


def residual_adaptivity(plot=False):
    """
    Make a run with adaptivity based on the residual.
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityResidual

    max_res = 1e-8
    custom_description = {'convergence_controllers': {}}
    custom_description['convergence_controllers'][AdaptivityResidual] = {
        'e_tol': max_res,
        'e_tol_low': max_res / 10,
    }
    stats, _, _ = run_piline(custom_description, num_procs=1)

    residual = get_sorted(stats, type='residual_post_step', recomputed=False)
    dt = get_sorted(stats, type='dt', recomputed=False)

    if plot:
        fig, ax = plt.subplots()
        dt_ax = ax.twinx()

        ax.plot([me[0] for me in residual], [me[1] for me in residual])
        dt_ax.plot([me[0] for me in dt], [me[1] for me in dt], color='black')
        plt.show()

    max_residual = max([me[1] for me in residual])
    assert max_residual < max_res, f'Max. allowed residual is {max_res:.2e}, but got {max_residual:.2e}!'
    dt_std = np.std([me[1] for me in dt])
    assert dt_std != 0, f'Expected the step size to change, but standard deviation is {dt_std:.2e}!'


def main():
    """
    Make a variety of tests to see if Hot Rod and Adaptivity work in serial as well as MSSDC.
    """
    generate_reference = False

    for use_adaptivity in [True, False]:
        custom_description = {'convergence_controllers': {}}
        if use_adaptivity:
            custom_description['convergence_controllers'][Adaptivity] = {
                'e_tol': 1e-7,
                'embedded_error_flavor': 'linearized',
            }

        for num_procs in [1, 4]:
            custom_description['convergence_controllers'][HotRod] = {'HotRod_tol': 1, 'no_storage': num_procs > 1}
            stats, _, _ = run_piline(custom_description, num_procs=num_procs)
            data = get_data(stats, recomputed=False)
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
                plt.close(sol_fig)
            check_solution(data, use_adaptivity, num_procs, generate_reference)
            plt.close(fig)


if __name__ == "__main__":
    residual_adaptivity()
    main()
