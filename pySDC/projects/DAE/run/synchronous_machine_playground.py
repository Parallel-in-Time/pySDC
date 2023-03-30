from pathlib import Path
import numpy as np
import pickle
import statistics

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.synchronous_machine import synchronous_machine_infinite_bus
from pySDC.projects.DAE.problems.synchronous_machine import synchronous_machine_pi_line
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats


def main():
    """
    A testing ground for the synchronous machine model 
    """
    #TODO: Run generator steady state for 10000 seconds. Try out different time step sizes. 
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-7
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3 # tollerance for implicit solver
    problem_params['nvars'] = 14

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [error_hook, approx_solution_hook]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = synchronous_machine_infinite_bus
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
    Tend = 10.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error
    err = get_sorted(stats, type='error_post_step', sortby='time')
    err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    print(f"Error is {err}")
    # assert np.isclose(err, 0.0, atol=1e-4), "Error too large."

    # store results
    sol = get_sorted(stats, type='approx_solution', sortby='time')
    sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    sol_data = np.array([[sol[j][1][i] for j in range(len(sol))] for i in range(problem_params['nvars'])])
    niter = filter_stats(stats, type='niter')
    niter = np.fromiter(niter.values(),int)
    
    data = dict()
    data['dt'] = sol_dt
    data['solution'] = sol_data
    data['niter']= round(statistics.mean(niter))
    pickle.dump(data, open("data/dae_conv_data.p", 'wb'))

    print("Done")


if __name__ == "__main__":
    main()
