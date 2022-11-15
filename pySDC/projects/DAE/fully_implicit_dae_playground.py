from pathlib import Path
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
from pySDC.projects.DAE.problems.simple_DAE import two_transistor_amplifier
from pySDC.projects.DAE.problems.simple_DAE import problematic_f
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.hooks.HookClass_approx_solution import approx_solution_hook
from pySDC.projects.DAE.hooks.HookClass_error import error_hook
from pySDC.helpers.stats_helper import get_sorted

def main():
    """
    A simple test program to run fully implicit SDC solver 
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 2 # need to work out exactly what this parameter does

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['log_to_file'] = True
    controller_params['fname'] = 'data/simple_dae_1.txt'
    controller_params['hook_class'] = approx_solution_hook # specialized hook class for more statistics and output

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
    Tend = 3.0 

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    # compute exact solution and compare
    # uex = P.u_exact(Tend)
    # err = abs(uex - uend)

    # out = 'Error after SDC iterations: %8.6e' % err
    # print(out)
    sol = get_sorted(stats, type='approx_solution_hook', sortby='time')
    data = [[sol[i][0], sol[i][1][0], sol[i][1][1]] for i in range(len(sol))]

    np.save("data/dae_data.npy", data)
    print("Done")

if __name__ == "__main__":
    main()