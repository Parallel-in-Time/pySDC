import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_step_size import LogStepSize

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI


class LogErrorEmbeddedEstimate(hooks):
    """
    Logs the data such as the numerical solution, the adapted step sizes by Adaptivity and the
    embedded error estimate.
    """

    def post_step(self, step, level_number):
        super(LogErrorEmbeddedEstimate, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_embedded',
            value=L.status.get('error_embedded_estimate'),
        )


class LogEvent(hooks):
    """
    Logs the problem dependent state function of the battery drain model.
    """

    def post_step(self, step, level_number):
        super(LogEvent, self).post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend[1] - P.V_ref[0],
        )


def generate_description(
    dt,
    problem,
    sweeper,
    num_nodes,
    hook_class,
    use_adaptivity,
    use_switch_estimator,
    problem_params,
    restol,
    maxiter,
    max_restarts=None,
    tol_event=1e-10,
):
    """
    Generate a description for the battery models for a controller run.

    Parameters
    ----------
    dt : float
        Time step for computation.
    problem : pySDC.core.Problem.ptype
        Problem class that wants to be simulated.
    sweeper : pySDC.core.Sweeper.sweeper
        Sweeper class for solving the problem class numerically.
    num_nodes : int
        Number of collocation nodes.
    hook_class : pySDC.core.Hooks
        Logged data for a problem.
    use_adaptivity : bool
        Flag if the adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    ncapacitors : int
        Number of capacitors used for the battery_model.
    alpha : float
        Multiple used for the initial conditions (problem_parameter).
    problem_params : dict
        Dictionary containing the problem parameters.
    restol : float
        Residual tolerance to terminate.
    maxiter : int
        Maximum number of iterations to be done.
    max_restarts : int, optional
        Maximum number of restarts per step.
    tol_event : float, optional
        Tolerance for switch estimation to terminate.

    Returns
    -------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1 if use_adaptivity else restol
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        switch_estimator_params['tol'] = tol_event
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

    if use_adaptivity:
        adaptivity_params = dict()
        adaptivity_params['e_tol'] = 1e-7
        convergence_controllers.update({Adaptivity: adaptivity_params})

    if max_restarts is not None:
        convergence_controllers[BasicRestartingNonMPI] = {
            'max_restarts': max_restarts,
            'crash_after_max_restarts': False,
        }

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    return description, controller_params


def controller_run(description, controller_params, use_adaptivity, use_switch_estimator, t0, Tend):
    """
    Executes a controller run for a problem defined in the description.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    use_adaptivity : bool
        Flag if the adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.

    Returns
    -------
    stats : dict
        Raw statistics from a controller run.
    """

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    problem = description['problem_class']
    sweeper = description['sweeper_class']

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/{}_{}_USE{}_USA{}.dat'.format(
        problem.__name__, sweeper.__name__, use_switch_estimator, use_adaptivity
    )
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    return stats


def run():
    """
    Executes the simulation for the battery model using two different sweepers and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 1e-2
    t0 = 0.0
    Tend = 0.3

    problem_classes = [battery, battery_implicit]
    sweeper_classes = [imex_1st_order, generic_implicit]
    num_nodes = 4
    restol = -1
    maxiter = 8

    ncapacitors = 1
    alpha = 1.2
    V_ref = np.array([1.0])
    C = np.array([1.0])

    problem_params = dict()
    problem_params['ncapacitors'] = ncapacitors
    problem_params['C'] = C
    problem_params['alpha'] = alpha
    problem_params['V_ref'] = V_ref

    max_restarts = 1
    recomputed = False
    use_switch_estimator = [True]
    use_adaptivity = [True]

    hook_class = [LogSolution, LogEvent, LogErrorEmbeddedEstimate, LogStepSize]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            for use_A in use_adaptivity:
                tol_event = 1e-10 if problem.__name__ == 'generic_implicit' else 1e-17

                description, controller_params = generate_description(
                    dt,
                    problem,
                    sweeper,
                    num_nodes,
                    hook_class,
                    use_A,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    max_restarts,
                    tol_event,
                )

                # Assertions
                proof_assertions_description(description, use_A, use_SE)

                proof_assertions_time(dt, Tend, V_ref, alpha)

                stats = controller_run(description, controller_params, use_A, use_SE, t0, Tend)

            check_solution(stats, dt, problem.__name__, use_A, use_SE)

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, use_A)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, use_adaptivity, cwd='./'):
    """
    Routine to plot the numerical solution of the model.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    problem : pySDC.core.Problem.ptype
        Problem class that wants to be simulated.
    sweeper : pySDC.core.Sweeper.sweeper
        Sweeper class for solving the problem class numerically.
    recomputed : bool
        Flag if the values after a restart are used or before.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    use_adaptivity : bool
        Flag if adaptivity wants to be used or not.
    cwd : str
        Current working directory.
    """

    f = open(cwd + 'data/{}_{}_USE{}_USA{}.dat'.format(problem, sweeper, use_switch_estimator, use_adaptivity), 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=recomputed)])
    vC = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    t = np.array([me[0] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Simulation of {} using {}'.format(problem, sweeper), fontsize=10)
    ax.plot(t, cL, label=r'$i_L$')
    ax.plot(t, vC, label=r'$v_C$')

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


def check_solution(stats, dt, problem, use_adaptivity, use_switch_estimator):
    """
    Function that checks the solution based on a hardcoded reference solution.
    Based on check_solution function from @brownbaerchen.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    dt : float
        Initial time step.
    problem : problem_class.__name__
        The problem_class that is numerically solved
    use_switch_estimator : bool
        Indicates if switch detection is used or not.
    use_adaptivity : bool
        Indicate if adaptivity is used or not.
    """

    data = get_data_dict(stats, use_adaptivity, use_switch_estimator)

    if problem == 'battery':
        if use_switch_estimator and use_adaptivity:
            msg = f'Error when using switch estimator and adaptivity for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5446532674094873,
                    'vC': 0.9999999999883544,
                    'dt': 0.01,
                    'e_em': 2.220446049250313e-16,
                    'state_function': -1.1645573394503117e-11,
                    'restarts': 3.0,
                    'sum_niters': 136.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.539386744746365,
                    'vC': 0.9999999710472945,
                    'dt': 0.005520873635314061,
                    'e_em': 2.220446049250313e-16,
                    'state_function': -2.8952705455331795e-08,
                    'restarts': 11.0,
                    'sum_niters': 264.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'dt': data['dt'][-1],
                'e_em': data['e_em'][-1],
                'state_function': data['state_function'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }
        elif use_switch_estimator and not use_adaptivity:
            msg = f'Error when using switch estimator for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5456190026495924,
                    'vC': 0.999166666941434,
                    'state_function': -0.0008333330585660326,
                    'restarts': 4.0,
                    'sum_niters': 296.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.5403849766797957,
                    'vC': 0.9999166666752302,
                    'state_function': -8.33333247698409e-05,
                    'restarts': 2.0,
                    'sum_niters': 2424.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'state_function': data['state_function'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

        elif not use_switch_estimator and use_adaptivity:
            msg = f'Error when using adaptivity for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.4433805288639916,
                    'vC': 0.90262388393713,
                    'dt': 0.18137307612335937,
                    'e_em': 2.7177844974524135e-09,
                    'restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.3994744179584864,
                    'vC': 0.9679037468770668,
                    'dt': 0.1701392217033212,
                    'e_em': 2.0992988458701234e-09,
                    'restarts': 0.0,
                    'sum_niters': 32.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'dt': data['dt'][-1],
                'e_em': data['e_em'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

    elif problem == 'battery_implicit':
        if use_switch_estimator and use_adaptivity:
            msg = f'Error when using switch estimator and adaptivity for battery_implicit for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5446675396652545,
                    'vC': 0.9999999999883541,
                    'dt': 0.01,
                    'e_em': 2.220446049250313e-16,
                    'state_function': -1.1645906461410505e-11,
                    'restarts': 3.0,
                    'sum_niters': 136.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.5393867447463223,
                    'vC': 0.9999999710472952,
                    'dt': 0.005520876908755634,
                    'e_em': 2.220446049250313e-16,
                    'state_function': -2.895270478919798e-08,
                    'restarts': 11.0,
                    'sum_niters': 264.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'dt': data['dt'][-1],
                'e_em': data['e_em'][-1],
                'state_function': data['state_function'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }
        elif use_switch_estimator and not use_adaptivity:
            msg = f'Error when using switch estimator for battery_implicit for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5456190026495138,
                    'vC': 0.9991666669414431,
                    'state_function': -0.0008333330585569287,
                    'restarts': 4.0,
                    'sum_niters': 296.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.5403849766797896,
                    'vC': 0.9999166666752302,
                    'state_function': -8.33333247698409e-05,
                    'restarts': 2.0,
                    'sum_niters': 2424.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'state_function': data['state_function'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

        elif not use_switch_estimator and use_adaptivity:
            msg = f'Error when using adaptivity for battery_implicit for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.4694087102919169,
                    'vC': 0.9026238839371302,
                    'dt': 0.18137307612335937,
                    'e_em': 2.3469713394952407e-09,
                    'restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'cL': 0.39947441811958956,
                    'vC': 0.9679037468770735,
                    'dt': 0.1701392217033212,
                    'e_em': 1.147640815712947e-09,
                    'restarts': 0.0,
                    'sum_niters': 32.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'dt': data['dt'][-1],
                'e_em': data['e_em'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

    for key in expected.keys():
        err_msg = f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'
        if key == 'cL':
            assert abs(expected[key] - got[key]) <= 1e-2, err_msg
        else:
            assert np.isclose(expected[key], got[key], rtol=1e-3), err_msg


def get_data_dict(stats, use_adaptivity, use_switch_estimator, recomputed=False):
    """
    Converts the statistics in a useful data dictionary so that it can be easily checked in the check_solution function.
    Based on @brownbaerchen's get_data function.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    use_adaptivity : bool
        Flag if adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    recomputed : bool
        Flag if the values after a restart are used or before.

    Returns
    -------
    data : dict
        Contains all information as the statistics dict.
    """

    data = dict()
    data['cL'] = np.array([me[1][0] for me in get_sorted(stats, type='u', sortby='time', recomputed=recomputed)])
    data['vC'] = np.array([me[1][1] for me in get_sorted(stats, type='u', sortby='time', recomputed=recomputed)])
    if use_adaptivity:
        data['dt'] = np.array(get_sorted(stats, type='dt', sortby='time', recomputed=recomputed))[:, 1]
        data['e_em'] = np.array(
            get_sorted(stats, type='error_embedded_estimate', sortby='time', recomputed=recomputed)
        )[:, 1]
    if use_switch_estimator:
        data['state_function'] = np.array(
            get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
        )[:, 1]
    if use_adaptivity or use_switch_estimator:
        data['restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])
    data['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

    return data


def get_recomputed(stats, type, sortby):
    """
    Function that filters statistics after a recomputation. It stores all value of a type before restart. If there are multiple values
    with same time point, it only stores the elements with unique times.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    type : str
        The type the be filtered.
    sortby : str
        String to specify which key to use for sorting.

    Returns
    -------
    sorted_list : list
        List of filtered statistics.
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
    Function to proof the assertions in the description.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    use_adaptivity : bool
        Flag if adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    """

    n = description['problem_params']['ncapacitors']
    assert (
        description['problem_params']['alpha'] > description['problem_params']['V_ref'][k] for k in range(n)
    ), 'Please set "alpha" greater than values of "V_ref"'
    assert type(description['problem_params']['V_ref']) == np.ndarray, '"V_ref" needs to be an np.ndarray'
    assert type(description['problem_params']['C']) == np.ndarray, '"C" needs to be an np.ndarray '
    assert (
        np.shape(description['problem_params']['V_ref'])[0] == n
    ), 'Number of reference values needs to be equal to number of condensators'
    assert (
        np.shape(description['problem_params']['C'])[0] == n
    ), 'Number of capacitance values needs to be equal to number of condensators'

    assert (
        description['problem_params']['V_ref'][k] > 0 for k in range(n)
    ), 'Please set values of "V_ref" greater than 0'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'

    if use_adaptivity:
        assert description['level_params']['restol'] == -1, "Please set restol to -1 or omit it"


def proof_assertions_time(dt, Tend, V_ref, alpha):
    """
    Function to proof the assertions regarding the time domain (in combination with the specific problem).

    Parameters
    ----------
    dt : float
        Time step for computation.
    Tend : float
        End time.
    V_ref : np.ndarray
        Reference values (problem parameter).
    alpha : float
        Multiple used for initial conditions (problem_parameter).
    """

    assert dt < Tend, "Time step is too large for the time domain!"

    assert (
        Tend == 0.3 and V_ref[0] == 1.0 and alpha == 1.2
    ), "Error! Do not use other parameters for V_ref != 1.0, alpha != 1.2, Tend != 0.3 due to hardcoded reference!"
    assert dt == 1e-2, "Error! Do not use another time step dt!= 1e-2!"


if __name__ == "__main__":
    run()
