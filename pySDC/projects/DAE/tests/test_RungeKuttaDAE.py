import pytest
import numpy as np

SWEEPER_NAMES = [
    'BackwardEulerDAE',
    'TrapezoidalRuleDAE',
    'DIRK43_2DAE',
    'EDIRK4DAE',
]


def get_sweeper(sweeper_name):
    """
    Retrieve a sweeper from a name. Thanks to @brownbaerchen!

    Parameters
    ----------
    sweeper_name : str
        Name of the sweeper as string.

    Returns
    -------
    pySDC.Sweeper.RungeKutta
        The sweeper.
    """
    import pySDC.projects.DAE.sweepers.RungeKuttaDAE as RK

    return eval(f'RK.{sweeper_name}')


def getTestSetup(problem, sweeper, hook_class):
    r"""
    Returns the description for the tests.

    Parameters
    ----------
    problem : pySDC.projects.DAE.misc.ptype_dae
        Problem class.
    sweeper : pySDC.projects.DAE.sweepers.RungeKuttaDAE
        Sweeper passed to the controller.
    hook_class : list
        Hook classes to log statistics such as errors.

    Returns
    -------
    description : dict
        Contains the parameters for one run.
    controller_params : dict
        Controller specific parameters.
    """

    level_params = {
        'restol': -1,
    }

    problem_params = {
        'newton_tol': 1e-14,
    }

    step_params = {
        'maxiter': 1,
    }

    controller_params = {
        'logger_level': 30,
        'hook_class': hook_class,
    }

    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'level_params': level_params,
        'step_params': step_params,
    }
    return description, controller_params


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def testOrderAccuracySemiExplicitIndexOne(sweeper_name):
    r"""
    In this test, the order of accuracy for different RK methods is
    tested for a semi-explicit differential algebraic equation (DAE)
    of index one. Here, order is tested in differential and algebraic
    part.
    """

    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.misc.HookClass_DAE import (
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    )
    from pySDC.helpers.stats_helper import get_sorted

    expectedOrderDiff = {
        'BackwardEulerDAE': 1,
        'TrapezoidalRuleDAE': 2,
        'DIRK43_2DAE': 3,
        'EDIRK4DAE': 4,
    }

    expectedOrderAlg = {
        'BackwardEulerDAE': 1,
        'TrapezoidalRuleDAE': 2,
        'DIRK43_2DAE': 3,
        'EDIRK4DAE': 4,
    }

    hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
    description, controller_params = getTestSetup(DiscontinuousTestDAE, get_sweeper(sweeper_name), hook_class)

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']

    t0, Tend = 1.0, 2.0
    dt_list = np.logspace(-1.7, -1.0, num=7)

    errorsDiff, errorsAlg = np.zeros(len(dt_list)), np.zeros(len(dt_list))
    for i, dt in enumerate(dt_list):
        level_params.update({'dt': dt})

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        errorsDiff[i] = max(
            np.array(get_sorted(stats, type='e_global_differential_post_step', sortby='time', recomputed=False))[:, 1]
        )
        errorsAlg[i] = max(
            np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time', recomputed=False))[:, 1]
        )

    orderDiff = np.mean(
        [
            np.log(errorsDiff[i] / errorsDiff[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
            for i in range(1, len(dt_list))
        ]
    )
    orderAlg = np.mean(
        [np.log(errorsAlg[i] / errorsAlg[i - 1]) / np.log(dt_list[i] / dt_list[i - 1]) for i in range(1, len(dt_list))]
    )

    assert np.isclose(
        orderDiff, expectedOrderDiff[sweeper_name], atol=1e0
    ), f"Expected order {expectedOrderDiff[sweeper_name]} in differential variable, got {orderDiff}"
    assert np.isclose(
        orderAlg, expectedOrderAlg[sweeper_name], atol=1e0
    ), f"Expected order {expectedOrderAlg[sweeper_name]} in algebraic variable, got {orderAlg}"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def testOrderAccuracySemiExplicitIndexTwo(sweeper_name):
    r"""
    In this test, the order of accuracy for different RK methods is
    tested for a semi-explicit differential algebraic equation (DAE)
    of index two. Here, order is tested in differential and algebraic
    part.

    Note that order reduction in the algebraic variable is expected.
    """

    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.misc.HookClass_DAE import (
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    )
    from pySDC.helpers.stats_helper import get_sorted

    expectedOrderDiff = {
        'BackwardEulerDAE': 1,
        'TrapezoidalRuleDAE': 2,
        'DIRK43_2DAE': 2,
        'EDIRK4DAE': 4,
    }

    expectedOrderAlg = {
        'BackwardEulerDAE': 1,
        'TrapezoidalRuleDAE': 2,
        'DIRK43_2DAE': 1,
        'EDIRK4DAE': 2,
    }

    hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
    description, controller_params = getTestSetup(simple_dae_1, get_sweeper(sweeper_name), hook_class)

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']

    t0, Tend = 0.0, 0.5
    dt_list = np.logspace(-1.7, -1.0, num=7)

    errorsDiff, errorsAlg = np.zeros(len(dt_list)), np.zeros(len(dt_list))
    for i, dt in enumerate(dt_list):
        level_params.update({'dt': dt})

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        errorsDiff[i] = max(
            np.array(get_sorted(stats, type='e_global_differential_post_step', sortby='time', recomputed=False))[:, 1]
        )
        errorsAlg[i] = max(
            np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time', recomputed=False))[:, 1]
        )

    orderDiff = np.mean(
        [
            np.log(errorsDiff[i] / errorsDiff[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
            for i in range(1, len(dt_list))
        ]
    )
    orderAlg = np.mean(
        [np.log(errorsAlg[i] / errorsAlg[i - 1]) / np.log(dt_list[i] / dt_list[i - 1]) for i in range(1, len(dt_list))]
    )

    assert np.isclose(
        orderDiff, expectedOrderDiff[sweeper_name], atol=1e0
    ), f"Expected order {expectedOrderDiff[sweeper_name]} in differential variable, got {orderDiff}"
    assert np.isclose(
        orderAlg, expectedOrderAlg[sweeper_name], atol=1e0
    ), f"Expected order {expectedOrderAlg[sweeper_name]} in algebraic variable, got {orderAlg}"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def testOrderAccuracyFullyImplicitIndexTwo(sweeper_name):
    r"""
    In this test, the order of accuracy for different RK methods is
    tested for a fully-implicit differential algebraic equation (DAE)
    of index two. Here, order is tested in all variables.

    Note that for index two problems order reduction is expected.
    """

    from pySDC.projects.DAE.problems.problematicF import problematic_f
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

    from pySDC.helpers.stats_helper import get_sorted

    expectedOrder = {
        'BackwardEulerDAE': 1,
        'TrapezoidalRuleDAE': 2,
        'DIRK43_2DAE': 1,
        'EDIRK4DAE': 2,
    }

    hook_class = [LogGlobalErrorPostStep]
    description, controller_params = getTestSetup(problematic_f, get_sweeper(sweeper_name), hook_class)

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']

    t0, Tend = 0.0, 2.0
    dt_list = np.logspace(-1.7, -1.0, num=7)

    errors = np.zeros(len(dt_list))
    for i, dt in enumerate(dt_list):
        level_params.update({'dt': dt})

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        errors[i] = max(np.array(get_sorted(stats, type='e_global_post_step', sortby='time', recomputed=False))[:, 1])

    order = np.mean(
        [np.log(errors[i] / errors[i - 1]) / np.log(dt_list[i] / dt_list[i - 1]) for i in range(1, len(dt_list))]
    )

    assert np.isclose(
        order, expectedOrder[sweeper_name], atol=1e0
    ), f"Expected order {expectedOrder[sweeper_name]} in differential variable, got {order}"
