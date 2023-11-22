from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import get_sorted
from mpi4py import MPI
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.playgrounds.compression.heat_compressed import heat_ND_compressed


def run_heat(Tend=1):
    # setup communicator
    # comm = MPI.COMM_WORLD if comm is None else comm

    # initialize problem parameters
    problem_params = {}
    problem_params["nu"] = 1
    problem_params["freq"] = (4, 4, 4)
    problem_params["order"] = 4
    problem_params["lintol"] = 1e-7
    problem_params["liniter"] = 99
    problem_params["solver_type"] = "CG"
    problem_params["nvars"] = (32, 32, 32)  # Have to be the same, Nx = Ny = Nz
    problem_params["bc"] = "periodic"

    # initialize level parameters
    level_params = {}
    level_params["restol"] = 5e-04
    level_params["dt"] = 1e-01
    level_params["nsweeps"] = 1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params["node_type"] = "LEGENDRE"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["QI"] = ["IE"]
    sweeper_params["QE"] = ["PIC"]
    sweeper_params["num_nodes"] = 3
    sweeper_params["initial_guess"] = "spread"

    # initialize step parameters
    step_params = {}
    step_params["maxiter"] = 50

    # initialize controller parameters
    controller_params = {}
    controller_params["logger_level"] = 15
    controller_params["hook_class"] = [LogSolution, LogGlobalErrorPostRun]

    # fill description dictionary for easy step instantiation
    description = {}
    # description['problem_class'] = heatNd_forced#     heat_ND_compressed
    description["problem_class"] = heat_ND_compressed
    description["problem_params"] = problem_params
    description["sweeper_class"] = imex_1st_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params

    # instantiate controller
    controller = controller_nonMPI(controller_params=controller_params, description=description, num_procs=1)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    return stats


def main():
    stats = run_heat(Tend=1)
    error = max([me[1] for me in get_sorted(stats, type="e_global_post_run")])
    # u = get_sorted(stats, type="u")
    print(error)


if __name__ == "__main__":
    main()
