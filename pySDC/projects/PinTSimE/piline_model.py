import matplotlib as mpl
import numpy as np
import dill
from scipy.integrate import solve_ivp

mpl.use('Agg')

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks


class log_data(hooks):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v1', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v2', value=L.uend[1])
        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='p3', value=L.uend[2])


def main():
    """
    A simple test program to do SDC/PFASST runs for the Piline model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = 1E-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = [3, 5]
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
    description['problem_class'] = piline                         # pass problem class
    description['problem_params'] = problem_params                # pass problem parameters
    description['sweeper_class'] = imex_1st_order                 # pass sweeper
    description['sweeper_params'] = sweeper_params                # pass sweeper parameters
    description['level_params'] = level_params                    # pass level parameters
    description['step_params'] = step_params                      # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh            # pass spatial transfer class

    assert 'errtol' not in description['step_params'].keys(), "No exact or reference solution known to compute error"

    # set time parameters
    t0 = 0.0
    Tend = 15
    
    num_procs = 8

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'piline.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('piline_out.txt', 'w')
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

    #plot_voltages(t0=t0, dt=level_params['dt'], Tend=Tend, uinit=uinit, problem_params=problem_params, reference_plotted=True)
    
    compute_ref_error(t0, level_params['dt'], Tend, uinit, problem_params)


def plot_voltages(t0=None, dt=None, Tend=None, uinit=None, problem_params=None, reference_plotted=False, cwd='./'):
    f = open(cwd + 'piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = get_sorted(stats, type='v1', sortby='time')
    v2 = get_sorted(stats, type='v2', sortby='time')
    p3 = get_sorted(stats, type='p3', sortby='time')

    times = [v[0] for v in v1]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in v1], linewidth=1, label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in v2], linewidth=1, label='$v_{C_2}$')
    ax.plot(times, [v[1] for v in p3], linewidth=1, label='$i_{L_\pi}$')
    
    if reference_plotted:
        ODE_Solvers = ['Radau', 'DOP853']
        linestyles = ['k--', 'r--']
        for style, ref_method in zip(linestyles, ODE_Solvers):
            v_ref = solve_ivp(piline_ODE, [t0, Tend], uinit, ref_method, args=problem_params.values(), dense_output=True, first_step=dt)
            
            v_sol = v_ref.sol(times)
            
            ax.plot(times, v_sol[0, :], style, label=ref_method)
            ax.plot(times, v_sol[1, :], style)
            ax.plot(times, v_sol[2, :], style)
    ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('piline_model_solution.png', dpi=300, bbox_inches='tight')
    
def compute_ref_error(t0, dt, Tend, uinit, problem_params, ref_method='DOP853', cwd='./'):
    """
        Routine to compute error between PFASST and a reference method
    """

    f = open(cwd + 'piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = get_sorted(stats, type='v1', sortby='time')
    v2 = get_sorted(stats, type='v2', sortby='time')
    p3 = get_sorted(stats, type='p3', sortby='time')
    
    v1_val = [v[1] for v in v1]
    v2_val = [v[1] for v in v2]
    p3_val = [v[1] for v in p3]

    times = [v[0] for v in v1]
    
    v_ref = solve_ivp(piline_ODE, [t0, Tend], uinit, ref_method, args=problem_params.values(), dense_output=True, first_step=dt)
    
    v_sol = v_ref.sol(times)
    
    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 3, figsize=(6, 2), sharex='col', sharey='row')
    ax[0].plot(times, v1_val-v_sol[0, :])
    ax[0].set_title('Ref.-Error of $v_{C_1}$')
    ax[0].set_yscale('log', base=10)
    ax[0].tick_params(axis='both')
    # ax[0].set_ylim(1e-9, 1e-1)
    ax[2].set_ylim(1e-14, 1e-1)
    ax[0].set_xlabel('Time')
    
    ax[1].plot(times, v2_val-v_sol[1, :])
    ax[1].set_title('Ref.-Error of $v_{C_2}$')
    ax[1].set_yscale('log', base=10)
    ax[1].tick_params(axis='both')
    # ax[1].set_ylim(1e-9, 1e-1)
    ax[2].set_ylim(1e-14, 1e-1)
    ax[1].set_xlabel('Time')
    
    ax[2].plot(times, p3_val-v_sol[2, :])
    ax[2].set_title('Ref.-Error of $i_{L_\pi}$')
    ax[2].set_yscale('log', base=10)
    ax[2].tick_params(axis='both')
    # ax[2].set_ylim(1e-9, 1e-1)
    ax[2].set_ylim(1e-14, 1e-1)
    ax[2].set_xlabel('Time')
    
    fig.savefig('piline_model_reference_error.png', dpi=300, bbox_inches='tight')
    
def piline_ODE(t, y, Vs, Rs, C1, Rpi, C2, Lpi, Rl):
    """
        Routine which defines the piline model problem as ODE for scipy ODE solvers
    """
    x1, x2, x3 = y
    dydt = [(-1/(Rs*C1))*x1 - (1/C1)*x3 + Vs/(Rs*C1), (-1/(Rl*C2))*x2 + (1/C2)*x3, (1/Lpi)*x1 - (1/Lpi)*x2 - (Rpi/Lpi)*x3]
    
    return dydt


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
