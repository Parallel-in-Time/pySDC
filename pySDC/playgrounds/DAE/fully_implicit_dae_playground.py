from pathlib import Path


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.simple_DAE import pendulum_2d
from pySDC.implementations.problem_classes.simple_DAE import simple_dae_1
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.fully_implicit_DAE import fully_implicit_DAE


def main():
    """
    A simple test program to run fully implicit SDC solver 
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-9  # tollerance for implicit solver
    problem_params['nvars'] = 3 # need to work out exactly what this parameter does

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 40

    # initialize controller parameters
    controller_params = dict()
    controller_params['log_to_file'] = True
    controller_params['fname'] = 'data/simple_dae_1.txt'

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.1
    Tend = 1.1  

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    out = 'Error after SDC iterations: %8.6e' % err
    print(out)
    print("Done")

if __name__ == "__main__":
    main()