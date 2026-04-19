import json
from pathlib import Path
import dolfin as df

from pySDC.projects.StroemungsRaum.problem_classes.HeatEquation_2D_FEniCS import fenics_heat2D_mass
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.hooks.log_solution import LogSolution


def setup(t0=None):
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
    dt = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0
    problem_params['c_nvars'] = 64
    problem_params['family'] = 'CG'
    problem_params['order'] = 2
    problem_params['c'] = 0.0

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = LogSolution

    description = dict()

    description['problem_class'] = fenics_heat2D_mass
    description['sweeper_class'] = imex_1st_order_mass

    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def run_simulation(description, controller_params, Tend):
    """
    Run the time integration for the heat equation problem.

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
        rel_err: float,
           relative final-time error.
    """
    # get initial time from description
    t0 = description['problem_params']['t0']

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # get exact solution at final time for error calculation
    uex = P.u_exact(Tend)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute relative error at final time
    rel_err = abs(uex - uend) / abs(uex)

    return P, stats, rel_err


def run_postprocessing(description, problem, stats, Tend):
    """
    Postprocess and store simulation results for visualization and analysis.

    Args:
        description: dict,
            pySDC description containing problem parameters.
        problem: Problem instance,
            Problem instance containing the final solution and other problem-related information.
        stats: dict,
            collected runtime statistics,
        Tend: float,
            Final simulation time.

    Returns: None
    """
    # Get the data directory
    import os

    path = f"{os.path.dirname(__file__)}/data/heat_equation/"

    # If it does not exist, create the 'data' directory at the specified path, including any necessary parent directories
    Path(path).mkdir(parents=True, exist_ok=True)

    # Save parameters
    parameters = description['problem_params']
    parameters.update(description['level_params'])
    parameters['Tend'] = Tend
    json.dump(parameters, open(path + "heat_equation_FEniCS_parameters.json", 'w'))

    # Create XDMF file for visualization output
    xdmffile_u = df.XDMFFile(path + "heat_equation_FEniCS_Temperature.xdmf")

    # Get the solution at every time step, sorted by time
    Solutions = get_sorted(stats, type='u', sortby='time')

    for i in range(len(Solutions)):
        time = Solutions[i][0]
        #
        un = Solutions[i][1]
        ux = problem.u_exact(time)
        #
        xdmffile_u.write_checkpoint(un.values, "un", time, df.XDMFFile.Encoding.HDF5, True)
        xdmffile_u.write_checkpoint(ux.values, "ux", time, df.XDMFFile.Encoding.HDF5, True)
        #
    xdmffile_u.close()

    return None


if __name__ == "__main__":

    t0 = 0.0
    Tend = 1.0

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # run the simulation and get the problem, stats and relative error
    problem, stats, rel_err = run_simulation(description, controller_params, Tend)
    print('The relative error at time ', Tend, 'is ', rel_err)

    # run postprocessing to save parameters and solution for visualization
    run_postprocessing(description, problem, stats, Tend)
