import pytest


@pytest.mark.base
@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_exactly_converged_step():
    """
    With `all_to_done`, a step that has already converged keeps iterating until the last one is done. Its update
    becomes exactly zero, which the iteration estimator used to divide by (and take the log of).
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.tutorial.step_8.C_iteration_estimator import setup_advection

    description, controller_params = setup_advection(dt=0.125, ndim=1, ml=True)
    controller_params.pop('hook_class')
    controller_params['all_to_done'] = True

    controller = controller_nonMPI(num_procs=8, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=1)

    assert [me[1] for me in get_sorted(stats, type='niter')] == [5] * 8
    assert abs(uend - P.u_exact(1)) < 1e-7
