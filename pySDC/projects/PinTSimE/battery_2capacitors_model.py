import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery_n_capacitors
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.battery_model import (
    controller_run,
    generate_description,
    get_recomputed,
    LogData,
    proof_assertions_description,
)
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


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
            type='state_function_1',
            value=L.uend[1] - P.V_ref[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function_2',
            value=L.uend[2] - P.V_ref[1],
        )


def run():
    """
    Executes the simulation for the battery model using the IMEX sweeper and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 4e-3
    t0 = 0.0
    Tend = 3.5

    problem_classes = [battery_n_capacitors]
    sweeper_classes = [imex_1st_order]
    num_nodes = 4
    restol = -1
    maxiter = 12

    ncapacitors = 2
    alpha = 5.0
    V_ref = np.array([1.0, 1.0])
    C = np.array([1.0, 1.0])

    problem_params = dict()
    problem_params['ncapacitors'] = ncapacitors
    problem_params['C'] = C
    problem_params['alpha'] = alpha
    problem_params['V_ref'] = V_ref

    recomputed = False
    use_switch_estimator = [True]

    hook_class = [LogData, LogEvent]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            description, controller_params = generate_description(
                dt,
                problem,
                sweeper,
                num_nodes,
                hook_class,
                False,
                use_SE,
                problem_params,
                restol,
                maxiter,
                1,
                1e-10,
            )

            # Assertions
            proof_assertions_description(description, False, use_SE)

            proof_assertions_time(dt, Tend, V_ref, alpha)

            stats = controller_run(description, controller_params, False, use_SE, t0, Tend)

            check_solution(stats, dt, use_SE)

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, False)


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
    vC1 = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=recomputed)])
    vC2 = np.array([me[1][2] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    t = np.array([me[0] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(t, cL, label='$i_L$')
    ax.plot(t, vC1, label='$v_{C_1}$')
    ax.plot(t, vC2, label='$v_{C_2}$')

    if use_switch_estimator:
        switches = get_recomputed(stats, type='switch', sortby='time')
        if recomputed is not None:
            assert len(switches) >= 2, f"Expected at least 2 switches, got {len(switches)}!"
        t_switches = [v[1] for v in switches]

        for i in range(len(t_switches)):
            ax.axvline(x=t_switches[i], linestyle='--', color='k', label='Switch {}'.format(i + 1))

    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/battery_2capacitors_model_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def check_solution(stats, dt, use_switch_estimator):
    """
    Function that checks the solution based on a hardcoded reference solution.
    Based on check_solution function from @brownbaerchen.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    dt : float
        Initial time step.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    """

    data = get_data_dict(stats, use_switch_estimator)

    if use_switch_estimator:
        msg = f'Error when using the switch estimator for battery_2capacitors for dt={dt:.1e}:'
        if dt == 1e-2:
            expected = {
                'cL': 1.1783297877614238,
                'vC1': 0.999999999996731,
                'vC2': 0.9999999999967274,
                'state_function_1': -3.2690516960087734e-12,
                'state_function_2': -3.272604409687574e-12,
                'restarts': 2.0,
                'sum_niters': 4236.0,
            }
        elif dt == 4e-1:
            expected = {
                'cL': 1.504055256840796,
                'vC1': 0.9999999999999989,
                'vC2': 0.999999999999998,
                'state_function_1': -1.1102230246251565e-15,
                'state_function_2': -1.9984014443252818e-15,
                'restarts': 8.0,
                'sum_niters': 252.0,
            }
        elif dt == 4e-2:
            expected = {
                'cL': 1.270787780316536,
                'vC1': 0.9999999999999913,
                'vC2': 0.9999999999999984,
                'state_function_1': -8.659739592076221e-15,
                'state_function_2': -1.5543122344752192e-15,
                'restarts': 8.0,
                'sum_niters': 1224.0,
            }
        elif dt == 4e-3:
            expected = {
                'cL': 1.1551903323589119,
                'vC1': 0.9999999999999931,
                'vC2': 0.9996666666666729,
                'state_function_1': -6.8833827526759706e-15,
                'state_function_2': -0.0003333333333270794,
                'restarts': 6.0,
                'sum_niters': 10620.0,
            }

    got = {
        'cL': data['cL'][-1],
        'vC1': data['vC1'][-1],
        'vC2': data['vC2'][-1],
        'state_function_1': data['state_function_1'][-1],
        'state_function_2': data['state_function_2'][-1],
        'restarts': data['restarts'],
        'sum_niters': data['sum_niters'],
    }

    for key in expected.keys():
        assert np.isclose(
            expected[key], got[key], rtol=1e-4
        ), f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'


def get_data_dict(stats, use_switch_estimator, recomputed=False):
    """
    Converts the statistics in a useful data dictionary so that it can be easily checked in the check_solution function.
    Based on @brownbaerchen's get_data function.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
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
    data['cL'] = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    data['vC1'] = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    data['vC2'] = np.array([me[1][2] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    data['state_function_1'] = np.array(get_sorted(stats, type='state_function_1', sortby='time', recomputed=False))[
        :, 1
    ]
    data['state_function_2'] = np.array(get_sorted(stats, type='state_function_2', sortby='time', recomputed=False))[
        :, 1
    ]
    data['restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])
    data['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

    return data


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
    alpha : np.float
        Multiple used for initial conditions (problem_parameter).
    """

    assert (
        Tend == 3.5 and V_ref[0] == 1.0 and V_ref[1] == 1.0 and alpha == 5.0
    ), "Error! Do not use other parameters for V_ref[:] != 1.0, alpha != 1.2, Tend != 0.3 due to hardcoded reference!"

    assert (
        dt == 1e-2 or dt == 4e-1 or dt == 4e-2 or dt == 4e-3
    ), "Error! Do not use other time steps dt != 4e-1 or dt != 4e-2 or dt != 4e-3 due to hardcoded references!"


if __name__ == "__main__":
    run()
