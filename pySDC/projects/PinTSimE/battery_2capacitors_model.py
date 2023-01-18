import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery_n_capacitors
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


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
            type='voltage C1',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C2',
            value=L.uend[2],
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


def main(use_switch_estimator=True):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model using 2 condensators

    Args:
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not

    Returns:
        description (dict): contains all information for a controller run
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = 1e-2

    assert level_params['dt'] == 1e-2, 'Error! Do not use the time step dt != 1e-2!'

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['ncondensators'] = 2
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = np.array([1.0, 1.0])
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 5.0
    problem_params['V_ref'] = np.array([1.0, 1.0])  # [V_ref1, V_ref2]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        convergence_controllers[SwitchEstimator] = switch_estimator_params

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery_n_capacitors  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, use_switch_estimator)

    # set time parameters
    t0 = 0.0
    Tend = 3.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_2condensators.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    recomputed = False

    check_solution(stats, level_params['dt'], use_switch_estimator)

    plot_voltages(description, recomputed, use_switch_estimator)

    return description


def plot_voltages(description, recomputed, use_switch_estimator, cwd='./'):
    """
    Routine to plot the numerical solution of the model

    Args:
        description(dict): contains all information for a controller run
        recomputed (bool): flag if the values after a restart are used or before
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        cwd: current working directory
    """

    f = open(cwd + 'data/battery_2condensators.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', recomputed=recomputed, sortby='time')
    vC1 = get_sorted(stats, type='voltage C1', recomputed=recomputed, sortby='time')
    vC2 = get_sorted(stats, type='voltage C2', recomputed=recomputed, sortby='time')

    times = [v[0] for v in cL]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in cL], label='$i_L$')
    ax.plot(times, [v[1] for v in vC1], label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in vC2], label='$v_{C_2}$')

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

    fig.savefig('data/battery_2condensators_model_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def check_solution(stats, dt, use_switch_estimator):
    """
    Function that checks the solution based on a hardcoded reference solution. Based on check_solution function from @brownbaerchen.

    Args:
        stats (dict): Raw statistics from a controller run
        dt (float): initial time step
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
    """

    data = get_data_dict(stats, use_switch_estimator)

    if use_switch_estimator:
        msg = f'Error when using the switch estimator for battery_2condensators for dt={dt:.1e}:'
        if dt == 1e-2:
            expected = {
                'cL': 1.2065280755094876,
                'vC1': 1.0094825899806945,
                'vC2': 1.0050052828742688,
                'switch1': 1.6094379124373626,
                'switch2': 3.209437912457051,
                'restarts': 2.0,
                'sum_niters': 1568,
            }
        elif dt == 4e-1:
            expected = {
                'cL': 1.1842780233981391,
                'vC1': 1.0094891393319418,
                'vC2': 1.00103823232433,
                'switch1': 1.6075867934844466,
                'switch2': 3.209437912436633,
                'restarts': 2.0,
                'sum_niters': 2000,
            }
        elif dt == 4e-2:
            expected = {
                'cL': 1.180493652021971,
                'vC1': 1.0094825917376264,
                'vC2': 1.0007713468084405,
                'switch1': 1.6094074085553605,
                'switch2': 3.209437912440314,
                'restarts': 2.0,
                'sum_niters': 2364,
            }
        elif dt == 4e-3:
            expected = {
                'cL': 1.1537529501025199,
                'vC1': 1.001438946726028,
                'vC2': 1.0004331625246141,
                'switch1': 1.6093728710270467,
                'switch2': 3.217437912434171,
                'restarts': 2.0,
                'sum_niters': 8920,
            }

    got = {
        'cL': data['cL'][-1],
        'vC1': data['vC1'][-1],
        'vC2': data['vC2'][-1],
        'switch1': data['switch1'],
        'switch2': data['switch2'],
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

    Args:
        stats (dict): Raw statistics from a controller run
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        recomputed (bool): flag if the values after a restart are used or before

    Return:
        data (dict): contains all information as the statistics dict
    """

    data = dict()
    data['cL'] = np.array(get_sorted(stats, type='current L', recomputed=recomputed, sortby='time'))[:, 1]
    data['vC1'] = np.array(get_sorted(stats, type='voltage C1', recomputed=recomputed, sortby='time'))[:, 1]
    data['vC2'] = np.array(get_sorted(stats, type='voltage C2', recomputed=recomputed, sortby='time'))[:, 1]
    data['switch1'] = np.array(get_recomputed(stats, type='switch', sortby='time'))[0, 1]
    data['switch2'] = np.array(get_recomputed(stats, type='switch', sortby='time'))[-1, 1]
    data['restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])
    data['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

    return data


def proof_assertions_description(description, use_switch_estimator):
    """
    Function to proof the assertions (function to get cleaner code)

    Args:
        description(dict): contains all information for a controller run
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
    """

    assert (
        description['problem_params']['alpha'] > description['problem_params']['V_ref'][0]
    ), 'Please set "alpha" greater than "V_ref1"'
    assert (
        description['problem_params']['alpha'] > description['problem_params']['V_ref'][1]
    ), 'Please set "alpha" greater than "V_ref2"'

    if description['problem_params']['ncondensators'] > 1:
        assert (
            type(description['problem_params']['V_ref']) == np.ndarray
        ), '"V_ref" needs to be an array (of type float)'
        assert (
            description['problem_params']['ncondensators'] == np.shape(description['problem_params']['V_ref'])[0]
        ), 'Number of reference values needs to be equal to number of condensators'
        assert (
            description['problem_params']['ncondensators'] == np.shape(description['problem_params']['C'])[0]
        ), 'Number of capacitance values needs to be equal to number of condensators'

    assert description['problem_params']['V_ref'][0] > 0, 'Please set "V_ref1" greater than 0'
    assert description['problem_params']['V_ref'][1] > 0, 'Please set "V_ref2" greater than 0'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'

    if use_switch_estimator:
        assert description['level_params']['restol'] == -1, "Please set restol to -1 or omit it"


if __name__ == "__main__":
    main()
