import numpy as np
from pathlib import Path

from pySDC.core.Errors import ParameterError

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit, battery_n_capacitors
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_step_size import LogStepSize
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI


class LogEventBattery(hooks):
    """
    Logs the problem dependent state function of the battery drain model.
    """

    def post_step(self, step, level_number):
        super(LogEventBattery, self).post_step(step, level_number)

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
            value=L.uend[1:] - P.V_ref[:],
        )


def generate_description(
    dt,
    problem,
    sweeper,
    num_nodes,
    quad_type,
    QI,
    hook_class,
    use_adaptivity,
    use_switch_estimator,
    problem_params,
    restol,
    maxiter,
    max_restarts=None,
    tol_event=1e-10,
    alpha=1.0,
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
    quad_type : str
        Type of quadrature nodes, e.g. 'LOBATTO' or 'RADAU-RIGHT'.
    QI : str
        Type of preconditioner used in SDC, e.g. 'IE' or 'LU'.
    hook_class : pySDC.core.Hooks
        Logged data for a problem.
    use_adaptivity : bool
        Flag if the adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
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
    alpha : float, optional
        Factor that indicates how the new step size in the Switch Estimator is reduced.

    Returns
    -------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    """

    # initialize level parameters
    level_params = {
        'restol': -1 if use_adaptivity else restol,
        'dt': dt,
    }
    if use_adaptivity:
        assert restol == -1, "Please set restol to -1 or omit it"

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': num_nodes,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': maxiter,
    }
    assert 'errtol' not in step_params.keys(), 'No exact solution known to compute error'

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': hook_class,
        'mssdc_jac': False,
    }

    # convergence controllers
    convergence_controllers = {}
    if use_switch_estimator:
        switch_estimator_params = {
            'tol': tol_event,
            'alpha': alpha,
        }
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})
    if use_adaptivity:
        adaptivity_params = {
            'e_tol': 1e-7,
        }
        convergence_controllers.update({Adaptivity: adaptivity_params})
    if max_restarts is not None:
        restarting_params = {
            'max_restarts': max_restarts,
            'crash_after_max_restarts': False,
        }
        convergence_controllers.update({BasicRestartingNonMPI: restarting_params})

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    return description, controller_params


def controller_run(description, controller_params, t0, Tend, exact_event_time_avail=None):
    """
    Executes a controller run for a problem defined in the description.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    exact_event_time_avail : bool, optional
        Indicates if exact event time of a problem is available.

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
    t_switch_exact = P.t_switch_exact if exact_event_time_avail is not None else None

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


def main():
    """
    Executes the simulation.
    """

    sweeper_params = {
        'num_nodes': 4,
        'quad_type': 'LOBATTO',
        'QI': 'IE',
    }

    hook_class = [LogSolution, LogEventBattery, LogEmbeddedErrorEstimate, LogStepSize]

    use_detection = [True]
    use_adaptivity = [True]

    handling_params = {
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
    }

    for problem, sweeper in zip([battery, battery_implicit], [imex_1st_order, generic_implicit]):

        for defaults in [False, True]:
            # ---- for hardcoded solutions defaults should match with parameters here ----
            if defaults:
                params_battery_1capacitor = {
                'ncapacitors': 1,
                }
            else:
                params_battery_1capacitor = {
                    'ncapacitors': 1,
                    'C': np.array([1.0]),
                    'alpha': 1.2,
                    'V_ref': np.array([1.0]),
                }    

            _ = run_simulation(
                problem=problem,
                problem_params=params_battery_1capacitor,
                sweeper=sweeper,
                sweeper_params=sweeper_params,
                use_adaptivity=use_adaptivity,
                use_detection=use_detection,
                handling_params=handling_params,
                hook_class=hook_class,
                interval=(0.0, 0.3),
                dt_list=[1e-2, 1e-3],
            )

    params_battery_2capacitors = {
        'ncapacitors': 2,
        'C': np.array([1.0, 1.0]),
        'alpha': 1.2,
        'V_ref': np.array([1.0, 1.0]),
    }

    _ = run_simulation(
        problem=battery_n_capacitors,
        problem_params=params_battery_2capacitors,
        sweeper=imex_1st_order,
        sweeper_params=sweeper_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        handling_params=handling_params,
        hook_class=hook_class,
        interval=(0.0, 0.5),
        dt_list=[1e-2, 1e-3],
    )


def run_simulation(problem, problem_params, sweeper, sweeper_params, use_adaptivity, use_detection, handling_params, hook_class, interval, dt_list):
    """
    Executes the simulation for the battery model using two different sweepers and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    prob_cls_name = problem.__name__
    unknowns = {
        'battery': ['i_L', 'v_C'],
        'battery_implicit': ['i_L', 'v_C'],
        'battery_n_capacitors': ['i_L', 'v_C1', 'v_C2'],
    }

    maxiter = 8
    restol = -1

    u_num = {}

    for dt in dt_list:
        u_num[dt] = {}

        for use_SE in use_detection:
            u_num[dt][use_SE] = {}

            for use_A in use_adaptivity:
                u_num[dt][use_SE][use_A] = {}

                description, controller_params = generate_description(
                    dt,
                    problem,
                    sweeper,
                    sweeper_params['num_nodes'],
                    sweeper_params['quad_type'],
                    sweeper_params['QI'],
                    hook_class,
                    use_adaptivity,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    handling_params['max_restarts'],
                    handling_params['tol_event'],
                    handling_params['alpha'],
                )

                stats, _ = controller_run(description, controller_params, interval[0], interval[-1])

                u_num[dt][use_SE][use_A] = get_data_dict(
                    stats, unknowns[prob_cls_name], use_A, use_SE, handling_params['recomputed']
                )

                plot_solution(u_num[dt][use_SE][use_A], prob_cls_name, use_A, use_SE)

                test_solution(u_num[dt][use_SE][use_A], prob_cls_name, dt, use_A, use_SE)

    return u_num


def plot_styling_stuff():
    """
    Returns plot stuff such as colors, line styles for making plots more pretty.
    """

    colors = {
        False: {
            False: 'dodgerblue',
            True: 'navy',
        },
        True: {
            False: 'linegreen',
            True: 'darkgreen',
        },
    }

    return colors


def plot_solution(u_num, prob_cls_name, use_adaptivity, use_detection):  # pragma: no cover
    r"""
    Plots the numerical solution for one simulation run.

    Parameters
    ----------
    u_num : dict
        Contains numerical solution with corresponding times for different problem_classes, and
        labels for different unknowns of the problem.
    prob_cls_name : str
        Name of the problem class to be plotted.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    """

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))

    unknowns = u_num['unknowns']
    for unknown in unknowns:
        ax.plot(u_num['t'], u_num[unknown], label=r"${}$".format(unknown))

    if use_detection:
        t_switches = u_num['t_switches']
        for i in range(len(t_switches)):
            ax.axvline(x=t_switches[i], linestyle='--', linewidth=0.8, color='r', label='Event {}'.format(i + 1))

    if use_adaptivity:
        dt_ax = ax.twinx()
        dt = u_num['dt']
        dt_ax.plot(dt[:, 0], dt[:, 1], linestyle='-', linewidth=0.8, color='k', label=r'$\Delta t$')
        dt_ax.set_ylabel(r'$\Delta t$', fontsize=16)
        dt_ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$u(t)$', fontsize=16)

    fig.savefig('data/{}_model_solution.png'.format(prob_cls_name), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def get_data_dict(stats, unknowns, use_adaptivity, use_detection, recomputed):
    """
    Extracts statistics and store it in a dictionary.

    Parameters
    ----------
    stats : dict
        Raw statistics of one simulation run.
    unknowns : list
        Unknowns of problem as string.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    use_detection : bool
        Indicates whether event detection is used or not.
    recomputed : bool
        Indicates if values after successfully steps are used or not.

    Returns
    -------
    res : dict
        Dictionary with extracted data separated with reasonable keys.
    """

    res =  {}

    # ---- numerical solution ----
    u_val = get_sorted(stats, type='u', sortby='time', recomputed=recomputed)
    res['t'] = np.array([item[0] for item in u_val])
    for i, label in enumerate(unknowns):
        res[label] = np.array([item[1][i] for item in u_val])

    res['unknowns'] = unknowns

    # ---- event time(s) found by event detection ----
    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No events found!'
        t_switches = [t[1] for t in switches]
        res['t_switches'] = t_switches

    h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
    h = np.array([np.abs(val[1]) for val in h_val])
    res['state_function'] = h

    # ---- embedded error and adapted step sizes----
    if use_adaptivity:
        res['e_em'] = np.array(
            get_sorted(stats, type='error_embedded_estimate', sortby='time', recomputed=recomputed)
        )
        res['dt'] = np.array(get_sorted(stats, type='dt', recomputed=recomputed))

    # ---- sum over restarts ----
    if use_adaptivity or use_detection:
        res['sum_restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])
    # ---- sum over all iterations ----
    res['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])
    return res


def test_solution(u_num, prob_cls_name, dt, use_adaptivity, use_detection):
    """
    Test for numerical solution if values satisfy hardcoded values.

    u_num : dict
        Contains the numerical solution together with event time found by event detection, step sizes adjusted
        via adaptivity and/or switch estimation.
    prob_cls_name : str
        Indicates which problem class is tested.
    dt : float
        (Initial) step sizes used for the simulation.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    use_detection : bool
        Indicates whether discontinuity handling is used in the simulation or not.
    """

    unknowns = u_num['unknowns']
    u_num_tmp = {unknown:u_num[unknown][-1] for unknown in unknowns}

    got = {}
    got = u_num_tmp

    if prob_cls_name == 'battery':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559718189012,
                    'v_C': 1.0053361988800296,
                    't_switches': [0.18232155679214296],
                    'dt': 0.11767844320785703,
                    'e_em': 7.811640223565064e-12,
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867578949986,
                    'v_C': 1.0000000000165197,
                    't_switches': [0.18232155677793654],
                    'dt': 0.015641173481932502,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559718189012,
                    'v_C': 1.0053361988800296,
                    't_switches': [0.18232155679214296],
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867578949986,
                    'v_C': 1.0000000000165197,
                    't_switches': [0.18232155677793654],
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4433805288639916,
                    'v_C': 0.90262388393713,
                    'dt': 0.18137307612335937,
                    'e_em': 2.7177844974524135e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.3994744179584864,
                    'v_C': 0.9679037468770668,
                    'dt': 0.1701392217033212,
                    'e_em': 2.0992988458701234e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4433805288639916,
                    'v_C': 0.90262388393713,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.3994744179584864,
                    'v_C': 0.9679037468770668,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })

    elif prob_cls_name == 'battery_implicit':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559717904407,
                    'v_C': 1.0053361988803866,
                    't_switches': [0.18232155679736195],
                    'dt': 0.11767844320263804,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867577837699,
                    'v_C': 1.0000000000250129,
                    't_switches': [0.1823215568036829],
                    'dt': 0.015641237833012522,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559717904407,
                    'v_C': 1.0053361988803866,
                    't_switches': [0.18232155679736195],
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867577837699,
                    'v_C': 1.0000000000250129,
                    't_switches': [0.1823215568036829],
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4694087102919169,
                    'v_C': 0.9026238839418407,
                    'dt': 0.18137307612335937,
                    'e_em': 2.3469713394952407e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.39947441811958956,
                    'v_C': 0.9679037468856341,
                    'dt': 0.1701392217033212,
                    'e_em': 1.147640815712947e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4694087102919169,
                    'v_C': 0.9026238839418407,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.39947441811958956,
                    'v_C': 0.9679037468856341,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })

    elif prob_cls_name == 'battery_n_capacitors':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.6244130166029733,
                    'v_C1': 0.999647921822499,
                    'v_C2': 1.0000000000714673,
                    't_switches': [0.18232155679216916, 0.3649951297770592],
                    'dt': 0.01,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 19.0,
                    'sum_niters': 400.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.6112496171462107,
                    'v_C1': 0.9996894956748836,
                    'v_C2': 1.0,
                    't_switches': [0.1823215567907929, 0.3649535697059346],
                    'dt': 0.07298158272977251,
                    'e_em': 2.703393064962256e-13,
                    'sum_restarts': 11.0,
                    'sum_niters': 216.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.6244130166029733,
                    'v_C1': 0.999647921822499,
                    'v_C2': 1.0000000000714673,
                    't_switches': [0.18232155679216916, 0.3649951297770592],
                    'sum_restarts': 19.0,
                    'sum_niters': 400.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.6112496171462107,
                    'v_C1': 0.9996894956748836,
                    'v_C2': 1.0,
                    't_switches': [0.1823215567907929, 0.3649535697059346],
                    'sum_restarts': 11.0,
                    'sum_niters': 216.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.15890544838473294,
                    'v_C1': 0.8806086293336285,
                    'v_C2': 0.9915019206727803,
                    'dt': 0.38137307612335936,
                    'e_em': 4.145817911194172e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.15422467570971707,
                    'v_C1': 0.8756872272783145,
                    'v_C2': 0.9971015415168025,
                    'dt': 0.3701392217033212,
                    'e_em': 3.6970297934146856e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.15890544838473294,
                    'v_C1': 0.8806086293336285,
                    'v_C2': 0.9915019206727803,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.15422467570971707,
                    'v_C1': 0.8756872272783145,
                    'v_C2': 0.9971015415168025,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })
    else:
        raise ParameterError(f"For {prob_cls_name} there is no test implemented here!")
    
    for key in expected.keys():
        if key == 't_switches':
            err_msg = f'{msg} Expected {key}={expected[key]}, got {key}={got[key]}'
            assert all(np.isclose(expected[key], got[key], atol=1e-4)) == True, err_msg
        else:
            err_msg = f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'
            assert np.isclose(expected[key], got[key], atol=1e-4), err_msg


def get_recomputed(stats, type, sortby='time'):
    """
    Function that filters statistics after a recomputation. It stores all value of a type before restart. If there are multiple values
    with same time point, it only stores the elements with unique times.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    type : str
        The type the be filtered.
    sortby : str, optional
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


if __name__ == "__main__":
    main()
