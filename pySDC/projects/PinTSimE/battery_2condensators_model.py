import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery_2Condensators import battery_2condensators
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
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
            value=L.uend[0]
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C1',
            value=L.uend[1]
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C2',
            value=L.uend[2]
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=1,
            initialize=0
        )


def main(use_switch_estimator=True):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model using 2 condensators
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-13
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C1'] = 1.0
    problem_params['C2'] = 1.0
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 5.0
    problem_params['V_ref'] = np.array([1.0, 1.0])  # [V_ref1, V_ref2]
    problem_params['set_switch'] = np.array([False, False], dtype=bool)
    problem_params['t_switch'] = np.zeros(np.shape(problem_params['V_ref'])[0])

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        convergence_controllers[SwitchEstimator] = switch_estimator_params

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery_2condensators  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

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

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('battery_2condensators_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        # print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])

    restarts = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
    print("Restarts for dt: ", level_params['dt'], " -- ", np.sum(restarts))

    assert np.mean(niters) <= 10, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    plot_voltages(description, use_switch_estimator)

    return np.mean(niters)


def plot_voltages(description, use_switch_estimator, cwd='./'):
    """
    Routine to plot the numerical solution of the model
    """

    f = open(cwd + 'battery_2condensators.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', sortby='time')
    vC1 = get_sorted(stats, type='voltage C1', sortby='time')
    vC2 = get_sorted(stats, type='voltage C2', sortby='time')

    times = [v[0] for v in cL]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in cL], label='$i_L$')
    ax.plot(times, [v[1] for v in vC1], label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in vC2], label='$v_{C_2}$')

    if use_switch_estimator:
        t_switch_plot = np.zeros(np.shape(description['problem_params']['t_switch'])[0])
        for i in range(np.shape(description['problem_params']['t_switch'])[0]):
            t_switch_plot[i] = description['problem_params']['t_switch'][i]

            ax.axvline(x=t_switch_plot[i], linestyle='--', color='k', label='Switch {}'.format(i + 1))

    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/battery_2condensators_model_solution.png', dpi=300, bbox_inches='tight')


def proof_assertions_description(description, problem_params):
    """
    Function to proof the assertions (function to get cleaner code)
    """

    assert problem_params['alpha'] > problem_params['V_ref'][0], 'Please set "alpha" greater than "V_ref1"'
    assert problem_params['alpha'] > problem_params['V_ref'][1], 'Please set "alpha" greater than "V_ref2"'

    assert problem_params['V_ref'][0] > 0, 'Please set "V_ref1" greater than 0'
    assert problem_params['V_ref'][1] > 0, 'Please set "V_ref2" greater than 0'

    assert type(problem_params['V_ref']) == np.ndarray, '"V_ref" needs to be an array (of type float)'
    assert not problem_params['set_switch'][0], 'First entry of "set_switch" needs to be False'
    assert not problem_params['set_switch'][1], 'Second entry of "set_switch" needs to be False'

    assert not type(problem_params['t_switch']) == float, '"t_switch" has to be an array with entry zero'

    assert problem_params['t_switch'][0] == 0, 'First entry of "t_switch" needs to be zero'
    assert problem_params['t_switch'][1] == 0, 'Second entry of "t_switch" needs to be zero'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'


if __name__ == "__main__":
    main()
