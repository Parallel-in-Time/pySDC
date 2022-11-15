import numpy as np
from matplotlib import pyplot as plt

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.hooks.HookClass_error import error_hook
from pySDC.helpers.stats_helper import get_sorted


def compute_convergence_data(cwd=''):
    """
    Routine to run the 1d acoustic-advection example with different orders

    Args:
        cwd (string): current working directory
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['newton_tol'] = 1e-9  # tollerance for implicit solver
    problem_params['nvars'] = 3 # need to work out exactly what this parameter does

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] = 30

    # initialize controller parameters
    controller_params = dict()
    controller_params['log_to_file'] = True
    controller_params['fname'] = 'data/simple_dae_1.txt'
    controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 1.0  

    num_samples = 30
    dt_vec = np.logspace(-2, -1, num=num_samples)
    error = np.zeros_like(dt_vec)
    for i in range(len(dt_vec)):

        level_params['dt'] = dt_vec[i]
        description['level_params'] = level_params

        # instantiate the controller
        controller = controller_nonMPI(
            num_procs=1, controller_params=controller_params, description=description
        )
        # get initial values
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        sol = get_sorted(stats, type='error_after_step', sortby='time')

        error[i] = np.linalg.norm([sol[i][1] for i in range(len(sol))], np.inf)

    results = np.row_stack((dt_vec, error))
    np.save("data/dae_data.npy", results)
    print("Done")


def plot_convergence_simple():
    data = np.load("data/dae_data.npy")

    # order_2 = np.logspace(-7, -3, num=len(data[0, :]))
    # order_3 = np.logspace(-13, -7, num=len(data[0, :]))
    # order_4 = np.logspace(-12, -4, num=len(data[0, :]))
    # order_5 = np.logspace(-17, -7, num=len(data[0, :]))
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.loglog(data[0, :], data[1, :], label="Error")
    # ax.loglog(data[0, :], order_2, label="Reference 2nd order", linestyle="--", linewidth=1, alpha=0.5)
    # ax.loglog(data[0, :], order_3, label="Reference 3rd order", linestyle="--", linewidth=1, alpha=0.5)
    # ax.loglog(data[0, :], order_4, label="Reference 4th order", linestyle="--", linewidth=1, alpha=0.5)
    # ax.loglog(data[0, :], order_5, label="Reference 5th order", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    # title='Convergence plot two stage implicit Runge-Kutta with Gauss nodes'
    ax.set(xlabel='dt', ylabel='max |error in x1|')
    ax.grid(visible=True)
    # fig.tight_layout()
    plt.legend()
    plt.show()
    # plt.savefig('results/conv_gauss_5.png')


if __name__ == "__main__":
    compute_convergence_data()
    plot_convergence_simple()
