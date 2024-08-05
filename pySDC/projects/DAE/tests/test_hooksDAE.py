import numpy as np
import pytest


@pytest.mark.base
@pytest.mark.parametrize('M', [2, 3, 4])
def testHookClassDiffAlgComps(M):
    """
    Test if the hook class returns the correct errors.
    """

    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.DAE.problems.discontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.misc.hooksDAE import (
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    )

    dt = 1e-2
    level_params = {
        'restol': 1e-13,
        'dt': dt,
    }

    problem_params = {
        'newton_tol': 1e-6,
    }

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': 'IE',
    }

    step_params = {
        'maxiter': 45,
    }

    controller_params = {
        'logger_level': 30,
        'hook_class': [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable],
    }

    description = {
        'problem_class': DiscontinuousTestDAE,
        'problem_params': problem_params,
        'sweeper_class': FullyImplicitDAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 1.0
    Tend = t0 + dt

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(Tend)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    errHookDiff = np.array(get_sorted(stats, type='e_global_differential_post_step', sortby='time'))[:, 1]
    errHookAlg = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))[:, 1]

    errRunDiff = abs(uex.diff[0] - uend.diff[0])
    errRunAlg = abs(uex.alg[0] - uend.alg[0])

    assert np.isclose(errHookDiff, errRunDiff), 'ERROR: Error in differential component does not match!'
    assert np.isclose(errHookAlg, errRunAlg), 'ERROR: Error in algebraic component does not match!'
