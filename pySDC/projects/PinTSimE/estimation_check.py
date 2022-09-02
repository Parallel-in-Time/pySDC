import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_model import log_data
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def run(dt, use_switch_estimator=True, V_ref=1):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = dt

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
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 5
    problem_params['V_ref'] = V_ref
    problem_params['set_switch'] = False
    problem_params['t_switch'] = False

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    switch_estimator_params = {}
    convergence_controllers = {SwitchEstimator: switch_estimator_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    assert problem_params['alpha'] > problem_params['V_ref'], 'Please set "alpha" greater than "V_ref"'
    assert problem_params['V_ref'] > 0, 'Please set "V_ref" greater than 0'
    assert level_params['dt'] <= 4E-1, 'Time step dt is too coarse, please set dt less than 4E-1'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'

    # set time parameters
    t0 = 0.0
    Tend = 2.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/battery.dat'
    fname = 'battery.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    f = open('battery_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])

    # assert np.mean(niters) <= 8
    assert np.mean(niters) <= 12, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return stats


def check(cwd='./'):
    """
        Routine to check the differences between using a switch estimator or not
    """

    V_ref = 1
    dt_list = [4E-1, 4E-2, 4E-3, 4E-4]
    # dt_list = [2e-2]
    use_switch_estimator = [True, False]
    for dt_item in dt_list:
        for item in use_switch_estimator:
            stats = run(dt=dt_item, use_switch_estimator=item, V_ref=V_ref)

            fname = 'battery_dt{}_USE{}.dat'.format(dt_item, item)
            f = open(fname, 'wb')
            dill.dump(stats, f)
            f.close()

            if item:
                restarts = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
                print(np.sum(restarts))

    val_switch_all = []
    diff_true_all = []
    diff_false_all_before = []
    diff_false_all_after = []
    for dt_item in dt_list:
        f1 = open(cwd + 'battery_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'battery_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch = get_sorted(stats_true, type='switch', sortby='time')
        t_switch = [v[0] for v in val_switch]
        vC_switch = [v[1] for v in val_switch]
        t_switch = t_switch[0]  # battery has only one single switch
        vC_switch = vC_switch[0]
        val_switch_all.append([t_switch, vC_switch])

        vC_true = get_sorted(stats_true, type='voltage C', recomputed=False, sortby='time')
        vC_false = get_sorted(stats_false, type='voltage C', sortby='time')

        diff_true = [v[1] - V_ref for v in vC_true]
        diff_false = [v[1] - V_ref for v in vC_false]

        times_true = [v[0] for v in vC_true]
        times_false = [v[0] for v in vC_false]
        for m in range(len(times_true)):
            if np.round(times_true[m], 15) == np.round(t_switch, 15):
                diff_true_all.append(diff_true[m])

        for m in range(1, len(times_false)):
            if times_false[m - 1] < t_switch < times_false[m]:
                diff_false_all_before.append(diff_false[m - 1])
                diff_false_all_after.append(diff_false[m])

        setup_mpl()
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax.set_title('Time evolution of $v_{C}-V_{ref}$')
        ax.plot(times_true, diff_true, label='SE=True', color='#ff7f0e')
        ax.plot(times_false, diff_false, label='SE=False', color='#1f77b4')
        ax.axvline(x=t_switch, linestyle='--', color='k', label='Switch')
        ax.legend(frameon=False, fontsize=10, loc='lower left')
        # ax.set_yticks(np.arange(-3e-2, 3e-2))
        # ax.set_xlim(t_switch-5e-1, 2.5)
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_xlabel('Time')

        fig.savefig('data/difference_estimation_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title("Difference $v_{C}-V_{ref}$")
    ax.plot(dt_list, diff_false_all_before, 'rs-', label='SE=False - before switch')
    ax.plot(dt_list, diff_false_all_after, 'bd--', label='SE=False - after switch')
    ax.plot(dt_list, diff_true_all, 'kd--', label='SE=True')
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.set_xticks(dt_list)
    ax.set_xticklabels(dt_list)
    ax.set_xscale('log', base=10)
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$\Delta t$")

    fig.savefig('data/diffs_estimation.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    check()
