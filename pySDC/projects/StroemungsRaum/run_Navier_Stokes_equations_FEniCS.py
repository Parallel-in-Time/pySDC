from pathlib import Path
import json

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_FEniCS import fenics_NSE_2D_mass
from pySDC.projects.StroemungsRaum.sweepers.imex_1st_order_mass_NSE import imex_1st_order_mass_NSE
from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_FEniCS import LogLiftDrag
from pySDC.helpers.stats_helper import get_sorted


def setup(t0=0):
    """
    Helper routine to set up parameters

    Args:
        t0: float,
            initial time

    Returns:
        description: dict,
            pySDC description dictionary containing problem and method parameters.
        controller_params: dict,
            Parameters for the pySDC controller.
    """
    # time step size
    dt = 1 / 1600

    # initialize level parameters
    level_params = dict()
    level_params['e_tol'] = 1e-9
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QE'] = ['PIC']
    sweeper_params['QI'] = ['LU']

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.001
    problem_params['t0'] = t0
    problem_params['family'] = 'CG'
    problem_params['order'] = 2

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = [LogLiftDrag]

    # Fill description dictionary
    description = dict()
    description['problem_class'] = fenics_NSE_2D_mass
    description['sweeper_class'] = imex_1st_order_mass_NSE
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def run_simulation(description, controller_params, Tend):
    """
    Run the time integration for the 2D Navier-Stokes equations.

     Args:
        description: dict,
            pySDC problem and method description.
        controller_params: dict,
            Parameters for the pySDC controller.
        Tend: float,
            Final simulation time.

    Returns:
        P: problem instance,
           Problem instance containing the final solution and other problem-related information.
        stats: dict,
           collected runtime statistics,
        uend: FEniCS function,
           Final solution at time Tend.
    """
    # get initial time from description
    t0 = description['problem_params']['t0']

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return P, stats, uend


def run_postprocessing(description, stats):
    """
    Postprocess and store simulation results for visualization and analysis.

    Args:
        description: dict,
            pySDC description containing problem parameters.
        problem: Problem instance,
            Problem instance containing the final solution and other problem-related information.
        stats: dict,
            collected runtime statistics,

    Returns: None
    """
    # get the data directory
    import os

    # create directory for storing results
    path = f"{os.path.dirname(__file__)}/data/navier_stokes/"
    Path(path).mkdir(parents=True, exist_ok=True)

    # extract lift and drag coefficients from the collected statistics
    lift = get_sorted(stats, type='lift_post_step', sortby='time')
    drag = get_sorted(stats, type='drag_post_step', sortby='time')

    # extract timing information
    timing = get_sorted(stats, type='timing_run', sortby='time')

    # save parameters
    parameters = description['problem_params']
    parameters.update(description['level_params'])
    parameters['Tend'] = lift[-1][0]
    parameters['timing'] = timing[0][1]
    with open(path + "Navier_Stokes_FEniCS_parameters.json", 'w') as f:
        json.dump(parameters, f)

    # save lift and drag coefficients
    with open(path + "Lift_Drag_Coefficients.txt", 'w') as f:
        for i in range(len(lift)):
            out = '%1.16f  %1.16f  %1.16f' % (lift[i][0], drag[i][1], lift[i][1])
            f.write(out + '\n')

    return None


if __name__ == "__main__":

    t0 = 3.125e-04
    Tend = 0.005

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # run the simulation and get the problem, stats and final solution
    problem, stats, uend = run_simulation(description, controller_params, Tend)

    # run postprocessing to save parameters and solutions for visualization
    run_postprocessing(description, stats)
