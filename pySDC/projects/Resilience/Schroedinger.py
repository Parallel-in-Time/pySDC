import numpy as np
from pathlib import Path
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import nonlinearschroedinger_imex
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft
from pySDC.projects.Resilience.hook import LogData, hook_collection


def run_Schroedinger(
    custom_description=None,
    num_procs=1,
    Tend=1.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    use_MPI=False,
    space_comm=None,
    **kwargs,
):
    """
    Run a Schroedinger problem with default parameters.

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

    space_comm = MPI.COMM_WORLD if space_comm is None else space_comm
    rank = space_comm.Get_rank()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1e-01 / 2
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = (128, 128)
    problem_params['spectral'] = False
    problem_params['comm'] = space_comm

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if rank == 0 else 99
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    description = dict()
    description['problem_params'] = problem_params
    description['problem_class'] = nonlinearschroedinger_imex
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if type(custom_description[k]) == dict:
                description[k] = {**description.get(k, {}), **custom_description.get(k, {})}
            else:
                description[k] = custom_description[k]

    # set time parameters
    t0 = 0.0

    # instantiate controller
    assert use_MPI == False, "MPI version in time not implemented"
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        nvars = [me / 2 for me in problem_params['nvars']]
        nvars[0] += 1

        rnd_args = {'iteration': 5, 'problem_pos': nvars, 'min_node': 1}
        args = {'time': 0.3, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, controller, Tend


def plot_solution(stats):  # pragma: no cover
    import matplotlib.pyplot as plt

    u = get_sorted(stats, type='u')
    fig, ax = plt.subplots()
    ax.imshow(np.abs(u[0][1]))
    plt.show()


def main():
    stats, _, _ = run_Schroedinger(space_comm=MPI.COMM_WORLD)
    plot_solution(stats)


if __name__ == "__main__":
    main()
