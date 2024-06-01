import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.misc.HookClass_DAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
)
from pySDC.helpers.stats_helper import get_sorted


def run(dt, num_nodes, use_MPI, semi_implicit, residual_type, index_case, initial_guess='spread', comm=None):
    r"""
    Prepares the controller with all the description needed. Here, the function decides to choose the correct sweeper
    for the test.

    Parameters
    ----------
    dt : float
        Time step size chosen for simulation.
    num_nodes : int
        Number of collocation nodes.
    use_MPI : bool
        If True, the MPI sweeper classes are used.
    semi_implicit : bool
        Modules are loaded either for fully-implicit case or semi-implicit case.
    residual_type : str
        Choose how to compute the residual.
    index_case : int
        Denotes the index case of a DAE to be tested here, can be either ``1`` or ``2``.
    initial_guess : str, optional
        Type of initial guess for simulation.
    comm : mpi4py.MPI.COMM_WORLD
        Communicator.
    """

    if not semi_implicit:
        if use_MPI:
            from pySDC.projects.DAE.sweepers.fully_implicit_DAE_MPI import fully_implicit_DAE_MPI as sweeper

        else:
            from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE as sweeper

    else:
        if use_MPI:
            from pySDC.projects.DAE.sweepers.SemiImplicitDAEMPI import SemiImplicitDAEMPI as sweeper

        else:
            from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE as sweeper

    if index_case == 1:
        from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE as problem

        t0 = 1.0
        Tend = 1.5

    elif index_case == 2:
        from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1 as problem

        t0 = 0.0
        Tend = 0.4

    else:
        raise NotImplementedError(f"DAE case of index {index_case} is not implemented!")

    # initialize level parameters
    level_params = {
        'restol': 1e-12,
        'residual_type': residual_type,
        'dt': dt,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-6,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
        'QI': 'MIN-SR-S',  # use a diagonal Q_Delta here!
        'initial_guess': initial_guess,
    }

    # check if number of processes requested matches with number of nodes
    if comm is not None:
        sweeper_params.update({'comm': comm})
        assert (
            sweeper_params['num_nodes'] == comm.Get_size()
        ), f"Number of nodes does not match with number of processes! Expected {sweeper_params['num_nodes']}, got {comm.Get_size()}!"

    # initialize step parameters
    step_params = {
        'maxiter': 20,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    controller.MS[0].levels[0].sweep.compute_end_point()

    residual = controller.MS[0].levels[0].status.residual

    return uend, residual, stats


def check_order(comm):
    num_nodes = comm.Get_size()
    use_MPI = True
    residual_type = 'full_abs'
    for semi_implicit in [False, True]:
        for index_case in [1, 2]:
            dt_list = np.logspace(-1.7, -1.0, num=5)

            errorsDiff, errorsAlg = np.zeros(len(dt_list)), np.zeros(len(dt_list))
            for i, dt in enumerate(dt_list):
                _, _, stats = run(
                    dt=dt,
                    num_nodes=num_nodes,
                    use_MPI=use_MPI,
                    semi_implicit=semi_implicit,
                    residual_type=residual_type,
                    index_case=index_case,
                    comm=comm,
                )

                errorsDiff[i] = max(
                    np.array(
                        get_sorted(stats, type='e_global_differential_post_step', sortby='time', recomputed=False)
                    )[:, 1]
                )
                errorsAlg[i] = max(
                    np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time', recomputed=False))[
                        :, 1
                    ]
                )

            # only process with index 0 should plot
            if comm.Get_rank() == 0:
                orderDiff = np.mean(
                    [
                        np.log(errorsDiff[i] / errorsDiff[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
                        for i in range(1, len(dt_list))
                    ]
                )
                orderAlg = np.mean(
                    [
                        np.log(errorsAlg[i] / errorsAlg[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
                        for i in range(1, len(dt_list))
                    ]
                )

                refOrderDiff = 2 * comm.Get_size() - 1
                refOrderAlg = 2 * comm.Get_size() - 1 if index_case == 1 else comm.Get_size()
                assert np.isclose(
                    orderDiff, refOrderDiff, atol=1e0
                ), f"Expected order {refOrderDiff} in differential variable, got {orderDiff}"
                assert np.isclose(
                    orderAlg, refOrderAlg, atol=1e0
                ), f"Expected order {refOrderAlg} in algebraic variable, got {orderAlg}"


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    check_order(comm)
