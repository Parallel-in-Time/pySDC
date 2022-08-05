import matplotlib as mpl
import numpy as np
import dill
from pathlib import Path

mpl.use('Agg')

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks


class log_data(hooks):
    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='v1',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='v2',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='p3',
            value=L.uend[2],
        )


def main():
    """
    A simple test program to do SDC/PFASST runs for the Piline model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 3
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 100.0
    problem_params['Rs'] = 1.0
    problem_params['C1'] = 1.0
    problem_params['Rpi'] = 0.2
    problem_params['C2'] = 1.0
    problem_params['Lpi'] = 1.0
    problem_params['Rl'] = 5.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    assert 'errtol' not in description['step_params'].keys(), "No exact or reference solution known to compute error"

    # set time parameters
    t0 = 0.0
    Tend = 20

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/piline.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('data/piline_out.txt', 'w')
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

    assert np.mean(niters) <= 10, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    plot_voltages()


def plot_voltages(cwd='./'):
    f = open(cwd + 'data/piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = get_sorted(stats, type='v1', sortby='time')
    v2 = get_sorted(stats, type='v2', sortby='time')
    p3 = get_sorted(stats, type='p3', sortby='time')

    times = [v[0] for v in v1]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in v1], linewidth=1, label=r'$v_{C_1}$')
    ax.plot(times, [v[1] for v in v2], linewidth=1, label=r'$v_{C_2}$')
    ax.plot(times, [v[1] for v in p3], linewidth=1, label=r'$i_{L_\pi}$')
    ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/piline_model_solution.png', dpi=300, bbox_inches='tight')


def setup_mpl(fontsize=8):
    plt_helper.setup_mpl(reset=True)

    style_options = {
        "font.family": "sans-serif",
        "font.serif": "Computer Modern Sans Serif",
        "font.sans-serif": "Computer Modern Sans Serif",
        "font.monospace": "Computer Modern Sans Serif",
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "legend.fontsize": 13,  # Make the legend/label fonts a little smaller
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,
        "lines.linewidth": 1,  # Make the plot lines a little smaller
    }

    mpl.rcParams.update(style_options)


if __name__ == "__main__":
    main()
