import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity


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
            type='current L',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
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
            value=L.status.get('error_embedded_estimate'),
        )


def main(dt, problem, sweeper, use_switch_estimator, use_adaptivity):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model

    Args:
        dt (float): time step for computation
        problem (problem_class.__name__): problem class that wants to be simulated
        sweeper (sweeper_class.__name__): sweeper class for solving the problem class numerically
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        use_adaptivity (bool): flag if the adaptivity wants to be used or not

    Returns:
        stats (dict): Raw statistics from a controller run
        description (dict): contains all information for a controller run
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_maxiter'] = 200
    problem_params['newton_tol'] = 1e-08
    problem_params['ncondensators'] = 1  # number of condensators
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1.0
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 1.2
    problem_params['V_ref'] = 1.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['mssdc_jac'] = False

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

    if use_adaptivity:
        adaptivity_params = dict()
        adaptivity_params['e_tol'] = 1e-7
        convergence_controllers.update({Adaptivity: adaptivity_params})

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator or use_adaptivity:
        description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0
    Tend = 0.3

    proof_assertions_description(description, use_adaptivity, use_switch_estimator)

    assert dt < Tend, "Time step is too large for the time domain!"

    assert (
        Tend == 0.3 and description['problem_params']['V_ref'] == 1.0 and description['problem_params']['alpha'] == 1.2
    ), "Error! Do not use other parameters for V_ref != 1.0, alpha != 1.2, Tend != 0.3 due to hardcoded reference!"
    assert description['level_params']['dt'] == 1e-2, "Error! Do not use another time step dt!= 1e-2!"

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_{}_USE{}_USA{}.dat'.format(sweeper.__name__, use_switch_estimator, use_adaptivity)
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    return stats, description


def run():
    """
    Executes the simulation for the battery model using two different sweepers and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 1e-2
    problem_classes = [battery, battery_implicit]
    sweeper_classes = [imex_1st_order, generic_implicit]
    recomputed = False
    use_switch_estimator = [True]
    use_adaptivity = [True]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            for use_A in use_adaptivity:
                stats, description = main(
                    dt=dt,
                    problem=problem,
                    sweeper=sweeper,
                    use_switch_estimator=use_SE,
                    use_adaptivity=use_A,
                )

            check_solution(stats, problem.__name__, use_adaptivity, use_switch_estimator)

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, use_A)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, use_adaptivity, cwd='./'):
    """
    Routine to plot the numerical solution of the model

    Args:
        description(dict): contains all information for a controller run
        problem (problem_class.__name__): problem class that wants to be simulated
        sweeper (sweeper_class.__name__): sweeper class for solving the problem class numerically
        recomputed (bool): flag if the values after a restart are used or before
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        cwd: current working directory
    """

    f = open(cwd + 'data/battery_{}_USE{}_USA{}.dat'.format(sweeper, use_switch_estimator, use_adaptivity), 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', recomputed=False, sortby='time')
    vC = get_sorted(stats, type='voltage C', recomputed=False, sortby='time')

    times = [v[0] for v in cL]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Simulation of {} using {}'.format(problem, sweeper), fontsize=10)
    ax.plot(times, [v[1] for v in cL], label=r'$i_L$')
    ax.plot(times, [v[1] for v in vC], label=r'$v_C$')

    if use_switch_estimator:
        switches = get_recomputed(stats, type='switch', sortby='time')

        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]
        ax.axvline(x=t_switch[-1], linestyle='--', linewidth=0.8, color='r', label='Switch')

    if use_adaptivity:
        dt = np.array(get_sorted(stats, type='dt', recomputed=False))

        dt_ax = ax.twinx()
        dt_ax.plot(dt[:, 0], dt[:, 1], linestyle='-', linewidth=0.8, color='k', label=r'$\Delta t$')
        dt_ax.set_ylabel(r'$\Delta t$', fontsize=8)
        dt_ax.legend(frameon=False, fontsize=8, loc='center right')

    ax.axhline(y=1.0, linestyle='--', linewidth=0.8, color='g', label='$V_{ref}$')

    ax.legend(frameon=False, fontsize=8, loc='upper right')

    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Energy', fontsize=8)

    fig.savefig('data/{}_model_solution_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def check_solution(stats, problem, use_adaptivity, use_switch_estimator):
    """
    Function that checks the solution based on a hardcoded reference solution. Based on check_solution function from @brownbaerchen.

    Args:
        stats (dict): Raw statistics from a controller run
        problem (problem_class.__name__): the problem_class that is numerically solved
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
    """

    data = get_data_dict(stats, use_adaptivity, use_switch_estimator)

    if use_switch_estimator and use_adaptivity:
        if problem == 'battery':
            msg = 'Error when using switch estimator and adaptivity for battery:'
            expected = {
                'cL': 0.5474500710994862,
                'vC': 1.0019332967173764,
                'dt': 0.011761752270047832,
                'e_em': 8.001793672107738e-10,
                'switches': 0.18232155791181945,
                'restarts': 3.0,
                'sum_niters': 44,
            }
        elif problem == 'battery_implicit':
            msg = 'Error when using switch estimator and adaptivity for battery_implicit:'
            expected = {
                'cL': 0.5424577937840791,
                'vC': 1.0001051105894005,
                'dt': 0.01,
                'e_em': 2.220446049250313e-16,
                'switches': 0.1822923488448394,
                'restarts': 6.0,
                'sum_niters': 60,
            }

    got = {
        'cL': data['cL'][-1],
        'vC': data['vC'][-1],
        'dt': data['dt'][-1],
        'e_em': data['e_em'][-1],
        'switches': data['switches'][-1],
        'restarts': data['restarts'],
        'sum_niters': data['sum_niters'],
    }

    for key in expected.keys():
        assert np.isclose(
            expected[key], got[key], rtol=1e-4
        ), f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'


def get_data_dict(stats, use_adaptivity=True, use_switch_estimator=True, recomputed=False):
    """
    Converts the statistics in a useful data dictionary so that it can be easily checked in the check_solution function.
    Based on @brownbaerchen's get_data function.

    Args:
        stats (dict): Raw statistics from a controller run
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        recomputed (bool): flag if the values after a restart are used or before

    Return:
        data (dict): contains all information as the statistics dict
    """

    data = dict()

    data['cL'] = np.array(get_sorted(stats, type='current L', recomputed=recomputed, sortby='time'))[:, 1]
    data['vC'] = np.array(get_sorted(stats, type='voltage C', recomputed=recomputed, sortby='time'))[:, 1]
    if use_adaptivity:
        data['dt'] = np.array(get_sorted(stats, type='dt', recomputed=recomputed, sortby='time'))[:, 1]
        data['e_em'] = np.array(
            get_sorted(stats, type='error_embedded_estimate', recomputed=recomputed, sortby='time')
        )[:, 1]
    if use_switch_estimator:
        data['switches'] = np.array(get_recomputed(stats, type='switch', sortby='time'))[:, 1]
    if use_adaptivity or use_switch_estimator:
        data['restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])
    data['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

    return data


def get_recomputed(stats, type, sortby):
    """
    Function that filters statistics after a recomputation. It stores all value of a type before restart. If there are multiple values
    with same time point, it only stores the elements with unique times.

    Args:
        stats (dict): Raw statistics from a controller run
        type (str): the type the be filtered
        sortby (str): string to specify which key to use for sorting

    Returns:
        sorted_list (list): list of filtered statistics
    """

    sorted_nested_list = []
    times_unique = np.unique([me[0] for me in get_sorted(stats, type=type)])
    filtered_list = [
        filter_stats(
            stats,
            time=t_unique,
            num_restarts=max([me.num_restarts for me in filter_stats(stats, type=type, time=t_unique).keys()]),
            type=type,
        )
        for t_unique in times_unique
    ]
    for item in filtered_list:
        sorted_nested_list.append(sort_stats(item, sortby=sortby))
    sorted_list = [item for sub_item in sorted_nested_list for item in sub_item]
    return sorted_list


def proof_assertions_description(description, use_adaptivity, use_switch_estimator):
    """
    Function to proof the assertions (function to get cleaner code)

    Args:
        description(dict): contains all information for a controller run
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
    """

    assert (
        description['problem_params']['alpha'] > description['problem_params']['V_ref']
    ), 'Please set "alpha" greater than "V_ref"'
    assert description['problem_params']['V_ref'] > 0, 'Please set "V_ref" greater than 0'
    assert type(description['problem_params']['V_ref']) == float, '"V_ref" needs to be of type float'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'

    if use_switch_estimator or use_adaptivity:
        assert description['level_params']['restol'] == -1, "For adaptivity, please set restol to -1 or omit it"


if __name__ == "__main__":
    run()
