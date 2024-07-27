from mpi4py import MPI
from pathlib import Path

from pySDC.projects.DAE.sweepers.semiImplicitDAEMPI import SemiImplicitDAEMPI
from pySDC.projects.DAE.problems.discontinuousTestDAE import DiscontinuousTestDAE
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.playgrounds.DAE.DiscontinuousTestDAE import plotSolution

from pySDC.implementations.hooks.log_solution import LogSolution


def run():
    r"""
    Routine to do a run using the semi-implicit SDC-DAE sweeper enabling parallelization across the nodes. The number of processes that is used to run this file is the number of collocation nodes used! When you run the script with the command 

    >>> mpiexec -n 3 python3 playground_MPI.py

    then 3 collocation nodes are used for SDC.
    """

    # This communicator is needed for the SDC-DAE sweeper
    comm = MPI.COMM_WORLD

    # initialize level parameters
    level_params = {
        'restol': 1e-12,
        'dt': 0.1,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-6,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': comm.Get_size(),
        'QI': 'MIN-SR-S',  # use a diagonal Q_Delta here!
        'initial_guess': 'spread',
        'comm': comm,
    }

    # check if number of processes requested matches with number of nodes
    assert sweeper_params['num_nodes'] == comm.Get_size(), f"Number of nodes does not match with number of processes!"

    # initialize step parameters
    step_params = {
        'maxiter': 20,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': DiscontinuousTestDAE,
        'problem_params': problem_params,
        'sweeper_class': SemiImplicitDAEMPI,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob

    t0 = 1.0
    Tend = 3.0
    
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)

    # only process with index 0 should plot
    if comm.Get_rank() == 0:
        file_name = 'data/solution_MPI.png'
        plotSolution(stats, file_name)


if __name__ == "__main__":
    run()
