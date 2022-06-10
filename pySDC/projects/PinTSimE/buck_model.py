import numpy as np
import dill

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.BuckConverter import buck_converter
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.PinTSimE.piline_model import log_data, setup_mpl
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to do SDC/PFASST runs for the buck converter model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12
    level_params['dt'] = 1E-5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['duty'] = 0.5  # duty cycle
    problem_params['fsw'] = 1e3  # switching freqency
    problem_params['Vs'] = 10.0
    problem_params['Rs'] = 0.5
    problem_params['C1'] = 1e-3
    problem_params['Rp'] = 0.01
    problem_params['L1'] = 1e-3
    problem_params['C2'] = 1e-3
    problem_params['Rl'] = 10

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = buck_converter   # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order   # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params      # pass level parameters
    description['step_params'] = step_params

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'duty' in description['problem_params'].keys(), 'Please supply "duty" in the problem parameters'
    assert 'fsw' in description['problem_params'].keys(), 'Please supply "fsw" in the problem parameters'

    assert 0 <= problem_params['duty'] <= 1, 'Please set "duty" greater than or equal to 0 and less than or equal to 1'

    # set time parameters
    t0 = 0.0
    Tend = 2e-2

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params,
                                   description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/buck.dat'
    fname = 'buck.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('buck_out.txt', 'w')
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

    assert np.mean(niters) <= 8, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    plot_voltages()


def plot_voltages(cwd='./'):
    f = open(cwd + 'buck.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = sort_stats(filter_stats(stats, type='v1'), sortby='time')
    v2 = sort_stats(filter_stats(stats, type='v2'), sortby='time')
    p3 = sort_stats(filter_stats(stats, type='p3'), sortby='time')

    times = [v[0] for v in v1]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in v1], linewidth=1, label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in v2], linewidth=1, label='$v_{C_2}$')
    ax.plot(times, [v[1] for v in p3], linewidth=1, label='$i_{L_\pi}$')
    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/buck_model_solution.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
