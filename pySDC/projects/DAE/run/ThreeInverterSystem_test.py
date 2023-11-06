import sys
sys.path.append("/home/jzh/data_jzh/pints_related/gits_2/pySDC")

from pathlib import Path
import numpy as np
import statistics
import pySDC.helpers.plot_helper as plt_helper
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.ThreeInverterSystem import ThreeInverterSystem
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
# from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
# from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats


def main():
    """
    A testing ground for the synchronous machine model
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-7
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 14

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogGlobalErrorPostStep]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = ThreeInverterSystem
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 0.5

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error (only available if reference solution was provided)
    # err = get_sorted(stats, type='error_post_step', sortby='time')
    # err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    # print(f"Error is {err}")

    # uend_ref = [
    #     8.30823565e-01,
    #     -4.02584174e-01,
    #     1.16966755e00,
    #     9.47592808e-01,
    #     -3.68076863e-01,
    #     -3.87492326e-01,
    #     -7.77837831e-01,
    #     -1.67347611e-01,
    #     1.34810867e00,
    #     5.46223705e-04,
    #     1.29690691e-02,
    #     -8.00823474e-02,
    #     3.10281509e-01,
    #     9.94039645e-01,
    # ]
    # err = np.linalg.norm(uend - uend_ref, np.inf)
    # assert np.isclose(err, 0, atol=1e-4), "Error too large."

    # store results
    sol = get_sorted(stats, type='approx_solution', sortby='time')
    sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    sol_data = np.array([[sol[j][1][i] for j in range(len(sol))] for i in range(problem_params['nvars'])])
    niter = filter_stats(stats, type='niter')
    niter = np.fromiter(niter.values(), int)

    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    # print([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time', recomputed=False)])
    i_c1 = np.array([me[0:2] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    v_cc1 = np.array([me[6:8] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    # data = dict()
    # data['dt'] = sol_dt
    # data['solution'] = sol_data
    # data['niter'] = round(statistics.mean(niter))
    # pickle.dump(data, open("data/dae_conv_data.p", 'wb'))
    file_name_suffix = "invSys"
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(t, i_c1[:, 0], label='ic1_0')
    ax.plot(t, i_c1[:, 1], label='ic1_1')
    ax.legend(loc='upper right', fontsize=10)
    # plt_helper.plt.show()
    fig.savefig(f'data/i_c1_{file_name_suffix}.png', dpi=300, bbox_inches='tight')


    print("Done")


if __name__ == "__main__":
    main()
