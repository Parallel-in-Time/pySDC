import pytest


def get_description(nsweeps, nlevels):
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    return {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': [[-1.0e-1 - 1j], [-1.0]][:nlevels], 'u0': 1.0},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'num_nodes': [3, 2][:nlevels], 'quad_type': 'RADAU-RIGHT', 'QI': 'IE'},
        'level_params': {'restol': 1e-10, 'dt': 0.1, 'nsweeps': nsweeps},
        'step_params': {'maxiter': 20},
        'space_transfer_class': mesh_to_mesh,
    }


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [1, 2])
@pytest.mark.parametrize('nlevels', [1, 2])
@pytest.mark.parametrize('mssdc_jac', [True, False])
@pytest.mark.parametrize('nsweeps', [1, 2])
def test_coarse_nsweeps_guard(num_procs, nlevels, mssdc_jac, nsweeps):
    """
    `it_coarse` sweeps the coarsest level exactly once, so nsweeps > 1 there must be rejected. That
    is every multi-level run, plus Gauss-like MSSDC, which is single-level but only exists for more
    than one step. Plain SDC (a single step) always routes through `it_fine` regardless of
    `mssdc_jac` and must keep working -- see https://github.com/Parallel-in-Time/pySDC/pull/668.
    """
    from pySDC.core.errors import ControllerError
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    reaches_it_coarse = nlevels > 1 or (num_procs > 1 and not mssdc_jac)
    controller_params = {'logger_level': 40, 'mssdc_jac': mssdc_jac}

    if nsweeps > 1 and reaches_it_coarse:
        with pytest.raises(ControllerError, match='multiple sweeps on coarsest level'):
            controller_nonMPI(num_procs, controller_params, get_description(nsweeps, nlevels))
        return

    controller = controller_nonMPI(num_procs, controller_params, get_description(nsweeps, nlevels))
    P = controller.MS[0].levels[0].prob
    Tend = num_procs * controller.MS[0].levels[0].params.dt
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    assert abs(uend - P.u_exact(Tend)) < 1e-8, 'run with an allowed nsweeps did not converge'


@pytest.mark.mpi4py
@pytest.mark.parametrize('nsweeps', [1, 2])
def test_coarse_nsweeps_guard_MPI(nsweeps):
    """Same for the MPI controller, on the single-step path that the guard used to reject."""
    from mpi4py import MPI
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

    controller = controller_MPI(
        {'logger_level': 40, 'mssdc_jac': False}, get_description(nsweeps, 1), comm=MPI.COMM_SELF
    )
    P = controller.S.levels[0].prob
    Tend = controller.S.levels[0].params.dt
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    assert abs(uend - P.u_exact(Tend)) < 1e-8, 'run with an allowed nsweeps did not converge'


if __name__ == '__main__':
    test_coarse_nsweeps_guard(1, 1, False, 2)
    test_coarse_nsweeps_guard_MPI(2)
