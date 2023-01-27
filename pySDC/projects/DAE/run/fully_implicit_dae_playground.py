from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import problematic_f
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted


def main():
    """
    A simple test program to see the fully implicit SDC solver in action
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 2  # need to work out exactly what this parameter does

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 40

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [approx_solution_hook, error_hook]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problematic_f
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
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error
    err = get_sorted(stats, type='error_post_step', sortby='time')
    err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    print(f"Error is {err}")
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."

    # store results
    sol = get_sorted(stats, type='approx_solution_hook', sortby='time')
    data = [[sol[i][0], sol[i][1][0], sol[i][1][1]] for i in range(len(sol))]

    pickle.dump(data, open("data/dae_conv_data.p", 'wb'))

    print("Done")


if __name__ == "__main__":
    main()
