import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
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
            type='u',
            value=L.uend,
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


def generate_description(
    dt,
    problem,
    sweeper,
    hook_class,
    use_adaptivity,
    use_switch_estimator,
    ncapacitors,
    alpha,
    V_ref,
    C,
    max_restarts=None,
):
    """
    Generate a description for the battery models for a controller run.
    Args:
        dt (float): time step for computation
        problem (pySDC.core.Problem.ptype): problem class that wants to be simulated
        sweeper (pySDC.core.Sweeper.sweeper): sweeper class for solving the problem class numerically
        hook_class (pySDC.core.Hooks): logged data for a problem
        use_adaptivity (bool): flag if the adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        ncapacitors (np.int): number of capacitors used for the battery_model
        alpha (np.float): Multiple used for the initial conditions (problem_parameter)
        V_ref (np.ndarray): Reference values for the capacitors (problem_parameter)
        C (np.ndarray): Capacitances (problem_parameter

    Returns:
        description (dict): contains all information for a controller run
        controller_params (dict): Parameters needed for a controller run
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    if problem == battery_implicit:
        problem_params['newton_maxiter'] = 200
        problem_params['newton_tol'] = 1e-08
    problem_params['ncapacitors'] = ncapacitors  # number of condensators
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = C
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = alpha
    problem_params['V_ref'] = V_ref

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
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
    Executes a controller run for a problem defined in the description

    Args:
        description (dict): contains all information for a controller run
        controller_params (dict): Parameters needed for a controller run
        use_adaptivity (bool): flag if the adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not

    Returns:
        stats (dict): Raw statistics from a controller run
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

    ncapacitors = 1
    alpha = 1.2
    V_ref = np.array([1.0])
    C = np.array([1.0])

    max_restarts = 1
    recomputed = False
    use_switch_estimator = [True]
    use_adaptivity = [True]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            for use_A in use_adaptivity:
                description, controller_params = generate_description(
                    dt, problem, sweeper, log_data, use_A, use_SE, ncapacitors, alpha, V_ref, C, max_restarts
                )

                # Assertions
                proof_assertions_description(description, use_A, use_SE)

                proof_assertions_time(dt, Tend, V_ref, alpha)

                stats = controller_run(description, controller_params, use_A, use_SE, t0, Tend)

            check_solution(stats, dt, problem.__name__, use_A, use_SE)

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, use_A)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, use_adaptivity, cwd='./'):
    """
    Routine to plot the numerical solution of the model

    Args:
        description(dict): contains all information for a controller run
        problem (pySDC.core.Problem.ptype): problem class that wants to be simulated
        sweeper (pySDC.core.Sweeper.sweeper): sweeper class for solving the problem class numerically
        recomputed (bool): flag if the values after a restart are used or before
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        cwd (str): current working directory
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
    Function that checks the solution based on a hardcoded reference solution. Based on check_solution function from @brownbaerchen.

    Args:
        stats (dict): Raw statistics from a controller run
        dt (float): initial time step
        problem (problem_class.__name__): the problem_class that is numerically solved
        use_switch_estimator (bool):
        use_adaptivity (bool):
    """

    data = get_data_dict(stats, use_adaptivity, use_switch_estimator)

    if problem == 'battery':
        if use_switch_estimator and use_adaptivity:
            msg = f'Error when using switch estimator and adaptivity for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5474500710994862,
                    'vC': 1.0019332967173764,
                    'dt': 0.011761752270047832,
                    'e_em': 8.001793672107738e-10,
                    'switches': 0.18232155791181945,
                    'restarts': 3.0,
                    'sum_niters': 44.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.5525783945667581,
                    'vC': 1.00001743462299,
                    'dt': 0.03550610373897258,
                    'e_em': 6.21240694442804e-08,
                    'switches': 0.18231603298272345,
                    'restarts': 4.0,
                    'sum_niters': 56.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5395601429161445,
                    'vC': 1.0000413761942089,
                    'dt': 0.028281271825675414,
                    'e_em': 2.5628611677319668e-08,
                    'switches': 0.18230920573953438,
                    'restarts': 3.0,
                    'sum_niters': 48.0,
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
        elif use_switch_estimator and not use_adaptivity:
            msg = f'Error when using switch estimator for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5495834172613568,
                    'vC': 1.000118710428906,
                    'switches': 0.1823188001399631,
                    'restarts': 1.0,
                    'sum_niters': 128.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.553775247309617,
                    'vC': 1.0010140038721593,
                    'switches': 0.1824302065533169,
                    'restarts': 1.0,
                    'sum_niters': 36.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5495840499078819,
                    'vC': 1.0001158309787614,
                    'switches': 0.18232183080236553,
                    'restarts': 1.0,
                    'sum_niters': 308.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'switches': data['switches'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

        elif not use_switch_estimator and use_adaptivity:
            msg = f'Error when using adaptivity for battery for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5401449976237487,
                    'vC': 0.9944656165121677,
                    'dt': 0.013143356036619536,
                    'e_em': 1.2462494369813726e-09,
                    'restarts': 3.0,
                    'sum_niters': 52.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.5966289599915113,
                    'vC': 0.9923148791604984,
                    'dt': 0.03564958366355817,
                    'e_em': 6.210964231812e-08,
                    'restarts': 1.0,
                    'sum_niters': 36.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5431613774808756,
                    'vC': 0.9934307674636834,
                    'dt': 0.022880524075396924,
                    'e_em': 1.1130212751453428e-08,
                    'restarts': 3.0,
                    'sum_niters': 52.0,
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
                    'cL': 0.5395401085152521,
                    'vC': 1.00003663985255,
                    'dt': 0.011465727118881608,
                    'e_em': 2.220446049250313e-16,
                    'switches': 0.18231044486762837,
                    'restarts': 4.0,
                    'sum_niters': 44.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.6717104472882885,
                    'vC': 1.0071670698947914,
                    'dt': 0.035896059229296486,
                    'e_em': 6.208836400567463e-08,
                    'switches': 0.18232158833761175,
                    'restarts': 3.0,
                    'sum_niters': 36.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5396216192241711,
                    'vC': 1.0000561014463172,
                    'dt': 0.009904645972832471,
                    'e_em': 2.220446049250313e-16,
                    'switches': 0.18230549652342606,
                    'restarts': 4.0,
                    'sum_niters': 44.0,
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
        elif use_switch_estimator and not use_adaptivity:
            msg = f'Error when using switch estimator for battery_implicit for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5495834122430945,
                    'vC': 1.000118715162845,
                    'switches': 0.18231880065636324,
                    'restarts': 1.0,
                    'sum_niters': 128.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.5537752525450169,
                    'vC': 1.0010140112484431,
                    'switches': 0.18243023230469263,
                    'restarts': 1.0,
                    'sum_niters': 36.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5495840604357269,
                    'vC': 1.0001158454740509,
                    'switches': 0.1823218812753008,
                    'restarts': 1.0,
                    'sum_niters': 308.0,
                }

            got = {
                'cL': data['cL'][-1],
                'vC': data['vC'][-1],
                'switches': data['switches'][-1],
                'restarts': data['restarts'],
                'sum_niters': data['sum_niters'],
            }

        elif not use_switch_estimator and use_adaptivity:
            msg = f'Error when using adaptivity for battery_implicit for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'cL': 0.5569818284195267,
                    'vC': 0.9846733115433628,
                    'dt': 0.01,
                    'e_em': 2.220446049250313e-16,
                    'restarts': 9.0,
                    'sum_niters': 88.0,
                }
            elif dt == 4e-2:
                expected = {
                    'cL': 0.5556563012729733,
                    'vC': 0.9930947318467772,
                    'dt': 0.035507110551631804,
                    'e_em': 6.2098696185231e-08,
                    'restarts': 6.0,
                    'sum_niters': 64.0,
                }
            elif dt == 4e-3:
                expected = {
                    'cL': 0.5401117929618637,
                    'vC': 0.9933888475391347,
                    'dt': 0.03176025170463925,
                    'e_em': 4.0386798239033794e-08,
                    'restarts': 8.0,
                    'sum_niters': 80.0,
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
        assert np.isclose(
            expected[key], got[key], rtol=1e-4
        ), f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'


def get_data_dict(stats, use_adaptivity, use_switch_estimator, recomputed=False):
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

    data['cL'] = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=recomputed, sortby='time')])
    data['vC'] = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=recomputed, sortby='time')])
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
    Function to proof the assertions in the description.

    Args:
        description(dict): contains all information for a controller run
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
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

    if use_switch_estimator or use_adaptivity:
        assert description['level_params']['restol'] == -1, "Please set restol to -1 or omit it"


def proof_assertions_time(dt, Tend, V_ref, alpha):
    """
    Function to proof the assertions regarding the time domain (in combination with the specific problem):

    Args:
        dt (float): time step for computation
        Tend (float): end time
        V_ref (np.ndarray): Reference values (problem parameter)
        alpha (np.float): Multiple used for initial conditions (problem_parameter)
    """

    assert dt < Tend, "Time step is too large for the time domain!"

    assert (
        Tend == 0.3 and V_ref[0] == 1.0 and alpha == 1.2
    ), "Error! Do not use other parameters for V_ref != 1.0, alpha != 1.2, Tend != 0.3 due to hardcoded reference!"
    assert dt == 1e-2, "Error! Do not use another time step dt!= 1e-2!"


if __name__ == "__main__":
    run()
