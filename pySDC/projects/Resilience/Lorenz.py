# script to run a Lorenz attractor problem
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.core.Errors import ProblemError
from pySDC.projects.Resilience.hook import LogData, hook_collection


def run_Lorenz(
    custom_description=None,
    num_procs=1,
    Tend=1.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    use_MPI=False,
    **kwargs,
):
    """
    Run a Lorenz attractor problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        custom_problem_params (dict): Overwrite presets
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    problem_params = {
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + hook_class if type(hook_class) == list else [hook_class]
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = LorenzAttractor
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

        # get initial values on finest level
        P = controller.S.levels[0].prob
        uinit = P.u_exact(t0)
    else:
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        raise NotImplementedError

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ProblemError:
        stats = controller.hooks.return_stats()

    return stats, controller, Tend


def plot_solution(stats):  # pragma: no cover
    """
    Plot the solution in 3D.

    Args:
        stats (dict): The stats object of the run

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    u = get_sorted(stats, type='u')
    ax.plot([me[1][0] for me in u], [me[1][1] for me in u], [me[1][2] for me in u])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def check_solution(stats, controller, thresh=5e-4):
    """
    Check if the global error solution wrt. a scipy reference solution is tolerable.
    This is also a check for the global error hook.

    Args:
        stats (dict): The stats object of the run
        controller (pySDC.Controller.controller): The controller
        thresh (float): Threshold for accepting the accuracy

    Returns:
        None
    """
    u = get_sorted(stats, type='u')
    u_exact = controller.MS[0].levels[0].prob.u_exact(t=u[-1][0])
    error = np.linalg.norm(u[-1][1] - u_exact, np.inf)
    error_hook = get_sorted(stats, type='e_global')[-1][1]

    assert error == error_hook, f'Expected errors to match, got {error:.2e} and {error_hook:.2e}!'
    assert error < thresh, f"Error too large, got e={error:.2e}"


def main(plotting=True):
    """
    Make a test run and see if the accuracy checks out.

    Args:
        plotting (bool): Plot the solution or not

    Returns:
        None
    """
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

    custom_description = {}
    custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-5}}
    custom_controller_params = {'logger_level': 30}
    stats, controller, _ = run_Lorenz(
        custom_description=custom_description,
        custom_controller_params=custom_controller_params,
        Tend=10,
        hook_class=[LogData, LogGlobalErrorPostRun],
    )
    check_solution(stats, controller, 5e-4)
    if plotting:  # pragma: no cover
        plot_solution(stats)


if __name__ == "__main__":
    main()
