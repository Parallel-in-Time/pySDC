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


def main(dt, problem, sweeper, use_switch_estimator, use_adaptivity):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
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
    controller_params['logger_level'] = 15
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

    proof_assertions_description(description, use_adaptivity, use_switch_estimator)

    # set time parameters
    t0 = 0.0
    Tend = 0.3

    assert dt < Tend, "Time step is too large for the time domain!"

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', recomputed=False, sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_{}_USE{}_USA{}.dat'.format(sweeper.__name__, use_switch_estimator, use_adaptivity)
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    f = open('data/battery_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])

    assert np.mean(niters) <= 4, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return description


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
                description = main(
                    dt=dt,
                    problem=problem,
                    sweeper=sweeper,
                    use_switch_estimator=use_SE,
                    use_adaptivity=use_A,
                )

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, use_A)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, use_adaptivity, cwd='./'):
    """
    Routine to plot the numerical solution of the model
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

def get_recomputed(stats, type, sortby):
    """
    Function that filters statistics after a recomputation
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
