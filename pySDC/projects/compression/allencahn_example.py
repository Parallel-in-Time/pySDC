from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import get_sorted
from mpi4py import MPI
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.compression.compressed_problems import (
    AllenCahn_MPIFFT_Compressed,
    allencahn_imex_timeforcing,
)
from pySDC.projects.compression.log_datatype_creations import LogDatatypeCreations
from pySDC.projects.compression.compression_convergence_controller import (
    Compression_Conv_Controller,
)


def run_AC(Tend=1):
    # setup communicator
    # comm = MPI.COMM_WORLD if comm is None else comm
    # initialize problem parameters
    problem_params = {}
    problem_params["eps"] = 0.04
    problem_params["radius"] = 0.25
    problem_params["spectral"] = False
    problem_params["dw"] = 0.0
    problem_params["L"] = 10
    problem_params["init_type"] = "circle_rand"
    problem_params["nvars"] = (128, 128)  # Have to be the same, Nx = Ny
    problem_params["comm"] = MPI.COMM_SELF

    convergence_controllers = {}
    convergence_controllers[Compression_Conv_Controller] = {"errBound": 1e-9}

    # initialize level parameters
    level_params = {}
    level_params["restol"] = 5e-07
    level_params["dt"] = 1e-03
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
    controller_params["hook_class"] = [
        LogSolution,
        # LogDatatypeCreations,
    ]

    # fill description dictionary for easy step instantiation
    description = {}
    # description['problem_class'] = AllenCahn_MPIFFT_Compressed
    description["problem_class"] = allencahn_imex_timeforcing
    description["problem_params"] = problem_params
    description["sweeper_class"] = imex_1st_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params
    description["convergence_controllers"] = convergence_controllers

    # instantiate controller
    controller = controller_nonMPI(
        controller_params=controller_params, description=description, num_procs=1
    )

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    return stats


def main():
    from pySDC.helpers.stats_helper import get_list_of_types, sort_stats, filter_stats

    stats = run_AC(Tend=0.002)
    print(get_list_of_types(stats))
    # print("filter_stats", filter_stats(stats, type="u"))
    # print("sort_stats", sort_stats(filter_stats(stats, type="u"), sortby="time"))
    u = get_sorted(stats, type="u")
    # print(u)
    import matplotlib.pyplot as plt

    # plt.plot([me[0] for me in u], [me[1] for me in u])
    # plt.show()
    plt.imshow(u[-1][1])
    plt.savefig("result_AC")


if __name__ == "__main__":
    main()
