import pytest


def run(sweeper_class, residual_type, comm=None):
    """A few steps of the forced heat equation with a diagonal preconditioner, returning end point and residual."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from mpi4py import MPI
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake

    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'MIN-SR-S', 'QE': 'PIC'}
    if comm is not None:
        sweeper_params['comm'] = comm
    description = {
        'problem_class': Heat1DForcedFiredrake,
        # the serial reference runs on every rank by itself
        'problem_params': {'n': 16, 'c': 1.0, 'comm': MPI.COMM_SELF if comm is None else comm.space_comm},
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'dt': 0.1, 'restol': -1.0, 'residual_type': residual_type},
        'step_params': {'maxiter': 4},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=0.3)
    return uend, controller.MS[0].levels[0].status.residual


@pytest.mark.firedrake
@pytest.mark.parallel(3)
@pytest.mark.parametrize('residual_type', ['full_abs', 'last_abs', 'full_rel', 'last_rel'])
def test_matches_the_serial_sweeper(residual_type):
    """One node per rank gives the serial mass sweeper's end point and residual, for every residual type."""
    import numpy as np
    from mpi4py import MPI
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass_MPI import imex_1st_order_mass_MPI

    ensemble = FiredrakeEnsembleCommunicator(MPI.COMM_WORLD, 1)
    u_par, res_par = run(imex_1st_order_mass_MPI, residual_type, comm=ensemble)
    u_ser, res_ser = run(imex_1st_order_mass, residual_type)

    assert np.abs(u_par.dat.data_ro - u_ser.dat.data_ro).max() < 1e-13
    # the residual is a small difference of O(1) terms, which the reduction adds in another order
    assert np.isclose(res_par, res_ser, rtol=0, atol=1e-15), (res_par, res_ser)
    assert res_par > 1e-12, 'four iterations should leave a residual to compare'
