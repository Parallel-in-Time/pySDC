import numpy as np
import scipy.io
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery_DrainCharge import battery_drain_charge
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
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

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='voltage CB', value=L.uend[0])


def main(use_switch_estimator=False, use_adaptivity=False):
    """
    A simple test program to do SDC/PFASST runs for the battery drain and charge model
    """

    pvData_dict = get_PV_data()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-7
    level_params['dt'] = 0.5

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
    problem_params['pvData'] = pvData_dict
    problem_params['Rs'] = 0.5
    problem_params['Rl'] = 1
    problem_params['CB'] = 1
    problem_params['RB'] = 1
    problem_params['R'] = 1
    problem_params['IR'] = 1
    problem_params['alpha'] = 3
    problem_params['set_switch'] = np.array([False], dtype=bool)
    problem_params['t_switch'] = np.zeros(1)

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

    if use_adaptivity:
        adaptivity_params = {'e_tol': 1e-7}
        convergence_controllers[Adaptivity] = adaptivity_params
        controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery_drain_charge  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator or use_adaptivity:
        description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0
    Tend = 25000

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/battery_drain_charge.dat'
    fname = 'battery_drain_charge.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('battery_out.txt', 'w')
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

    assert np.mean(niters) <= 7, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    plot_voltage(description, use_switch_estimator, use_adaptivity)

    return np.mean(niters)


def get_PV_data():
    """
    Function that converts the PV data into a dictionary
    """

    pvData_mat = scipy.io.loadmat(r"PV_Power_profiles.mat")  # dictionary
    pvData = pvData_mat['S']  # type np.array
    pvData_dict = dict(enumerate(pvData.flatten(), 1))

    return pvData_dict


def plot_voltage(description, use_switch_estimator, use_adaptivity, cwd='./'):
    """
        Routine to plot the numerical solution of the model
    """

    f = open(cwd + 'battery_drain_charge.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    vCB = get_sorted(stats, type='voltage CB', sortby='time')

    times = [v[0] for v in vCB]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in vCB], label='$v_{C_{B}}$')

    if use_switch_estimator:
        val_switch = get_sorted(stats, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        ax.axvline(x=t_switch, linestyle='--', color='k', label='Switch')

    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('battery_drain_charge_model_solution.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
