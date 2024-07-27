from pathlib import Path
import numpy as np
import pickle
import statistics

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.synchronousMachine import SynchronousMachineInfiniteBus
from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE
from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostStepDifferentialVariable
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats
from pySDC.implementations.hooks.log_solution import LogSolution


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
    problem_params['newton_tol'] = 1e-3

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogGlobalErrorPostStepDifferentialVariable, LogSolution]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = SynchronousMachineInfiniteBus
    description['problem_params'] = problem_params
    description['sweeper_class'] = FullyImplicitDAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error (only available if reference solution was provided)
    # err = get_sorted(stats, type='e_global_differential_post_step', sortby='time')
    # err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    # print(f"Error is {err}")

    uend_ref = P.dtype_u(P.init)
    uend_ref.diff[:8] = (
        8.30823565e-01,
        -4.02584174e-01,
        1.16966755e00,
        9.47592808e-01,
        -3.68076863e-01,
        -3.87492326e-01,
        3.10281509e-01,
        9.94039645e-01,
    )
    uend_ref.alg[:6] = (
        -7.77837831e-01,
        -1.67347611e-01,
        1.34810867e00,
        5.46223705e-04,
        1.29690691e-02,
        -8.00823474e-02,
    )
    err = abs(uend.diff - uend_ref.diff)
    assert np.isclose(err, 0, atol=1e-4), "Error too large."

    # store results
    sol = get_sorted(stats, type='u', sortby='time')
    sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    sol_data = np.array(
        [
            [(sol[j][1].diff[id], sol[j][1].alg[ia]) for j in range(len(sol))]
            for id, ia in zip(range(len(uend.diff)), range(len(uend.alg)))
        ]
    )
    niter = filter_stats(stats, type='niter')
    niter = np.fromiter(niter.values(), int)

    data = dict()
    data['dt'] = sol_dt
    data['solution'] = sol_data
    data['niter'] = round(statistics.mean(niter))
    pickle.dump(data, open("data/dae_conv_data.p", 'wb'))

    print("Done")


if __name__ == "__main__":
    main()
