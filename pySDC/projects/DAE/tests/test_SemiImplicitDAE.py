import pytest
import numpy as np


def getTestSetup(problem, sweeper, hook_class):
    r"""
    Returns the description for the tests.

    Parameters
    ----------
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
        'restol': 1e-13,
    }

    problem_params = {
        'newton_tol': 1e-6,
    }

    step_params = {
        'maxiter': 60,
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
@pytest.mark.parametrize('initial_guess', ['spread', 'zero', 'random'])
def testPredict(initial_guess):
    r"""
    In this test the predict function of the sweeper is tested.
    """

    from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh
    from pySDC.core.Step import step

    description, _ = getTestSetup(DiscontinuousTestDAE, SemiImplicitDAE, [])

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
        'initial_guess': initial_guess,
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']
    level_params.update({'dt': 0.1})
    description.update({'level_params': level_params})

    S = step(description=description)
    L = S.levels[0]
    P = L.prob

    assert isinstance(L.sweep, SemiImplicitDAE), "Sweeper cannot instantiate an object of type SemiImplicitDAE!"

    L.status.time = 1.0
    L.u[0] = P.u_exact(L.time)

    L.sweep.predict()

    assert isinstance(L.u[0], DAEMesh), "Initial condition u0 is not of type DAEMesh!"
    assert isinstance(L.f[0], DAEMesh), "Initial condition f0 is not of type DAEMesh!"

    assert np.allclose(L.f[0], 0.0), "Gradient at starting time needs to be initialised as zero!"
    if initial_guess == 'spread':
        uSpread = [L.u[m] for m in range(1, L.sweep.coll.num_nodes)]
        fSpread = [L.f[m] for m in range(1, L.sweep.coll.num_nodes)]
        assert np.allclose(uSpread, L.u[0]), "Initial condition u0 is not spreaded!"
        assert np.allclose(fSpread, L.f[0]), "Gradient needs to be spreaded as zero!"
    elif initial_guess == 'zero':
        uZero = [L.u[m] for m in range(1, L.sweep.coll.num_nodes)]
        fZero = [L.f[m] for m in range(1, L.sweep.coll.num_nodes)]
        assert np.allclose(uZero, 0.0), "Initial condition u0 is not spreaded!"
        assert np.allclose(fZero, L.f[0]), "Gradient needs to be spreaded as zero!"
    elif initial_guess == 'random':
        uRandom = [L.u[m] for m in range(1, L.sweep.coll.num_nodes)]
        fRandom = [L.f[m] for m in range(1, L.sweep.coll.num_nodes)]
        assert all(abs(uRandomItem) > 0.0 for uRandomItem in uRandom), "Initial condition u0 is not spreaded!"
        assert all(abs(fRandomItem) > 0.0 for fRandomItem in fRandom), "Gradient needs to be spreaded as zero!"


@pytest.mark.base
@pytest.mark.parametrize('residual_type', ['full_abs', 'last_abs', 'full_rel', 'last_rel', 'else'])
def testComputeResidual(residual_type):
    r"""
    In this test the predict function of the sweeper is tested.
    """

    from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.core.Step import step
    from pySDC.core.Errors import ParameterError

    description, _ = getTestSetup(DiscontinuousTestDAE, SemiImplicitDAE, [])

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
        'initial_guess': 'spread',
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']
    level_params.update({'dt': 0.1})
    level_params.update({'residual_type': residual_type})
    description.update({'level_params': level_params})

    S = step(description=description)
    L = S.levels[0]
    P = L.prob

    L.status.time = 1.0
    L.u[0] = P.u_exact(L.time)
    L.sweep.predict()
    if residual_type == 'else':
        with pytest.raises(ParameterError):
            L.sweep.compute_residual()
    else:
        L.sweep.compute_residual()

    uRef = P.u_exact(L.time)
    duRef = P.dtype_f(P.init)

    resNormRef = []
    for m in range(L.sweep.coll.num_nodes):
        # use abs function from data type here
        resNormRef.append(abs(P.eval_f(uRef, duRef, L.time + L.dt * L.sweep.coll.nodes[m])))

    if residual_type == 'full_abs':
        assert L.status.residual == max(resNormRef)
    elif residual_type == 'last_abs':
        assert L.status.residual == resNormRef[-1]
    elif residual_type == 'full_rel':
        assert L.status.residual == max(resNormRef) / abs(uRef)
    elif residual_type == 'last_rel':
        assert L.status.residual == resNormRef[-1] / abs(uRef)


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'RADAU-LEFT'])
def testComputeEndpoint(quad_type):
    r"""
    In this test the predict function of the sweeper is tested.
    """

    from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.core.Step import step
    from pySDC.core.Errors import ParameterError

    description, _ = getTestSetup(DiscontinuousTestDAE, SemiImplicitDAE, [])

    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': 2,
        'QI': 'IE',
        'initial_guess': 'spread',
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']
    level_params.update({'dt': 0.1})
    description.update({'level_params': level_params})

    if quad_type == 'RADAU-LEFT':
        with pytest.raises(ParameterError):
            S = step(description=description)
            with pytest.raises(NotImplementedError):
                S.levels[0].sweep.compute_end_point()
    else:
        S = step(description=description)

        L = S.levels[0]
        P = L.prob

        L.status.time = 1.0
        L.u[0] = P.u_exact(L.time)
        L.sweep.predict()

        assert isinstance(L.uend, type(None)), "u at end node is not of NoneType!"

        L.sweep.compute_end_point()

        assert np.isclose(L.u[-1], L.uend), "Endpoint is not computed correctly!"


@pytest.mark.base
@pytest.mark.parametrize('M', [2, 3])
def testCompareResults(M):
    r"""
    Test checks whether the results of the ``fully_implicit_DAE`` sweeper matches
    with the ``SemiImplicitDAE`` version.
    """

    from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    descrSI, controller_params = getTestSetup(DiscontinuousTestDAE, SemiImplicitDAE, [])
    descrFI, _ = getTestSetup(DiscontinuousTestDAE, fully_implicit_DAE, [])

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': 'IE',
    }
    descrSI.update({'sweeper_params': sweeper_params})
    descrFI.update({'sweeper_params': sweeper_params})

    level_paramsSI = descrSI['level_params']
    level_paramsSI.update({'dt': 0.1})
    descrSI.update({'level_params': level_paramsSI})

    level_paramsFI = descrFI['level_params']
    level_paramsFI.update({'dt': 0.1})
    descrFI.update({'level_params': level_paramsFI})

    t0 = 1.0
    Tend = 1.1

    controllerSI = controller_nonMPI(num_procs=1, controller_params=controller_params, description=descrSI)
    controllerFI = controller_nonMPI(num_procs=1, controller_params=controller_params, description=descrFI)

    P = controllerSI.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uendSI, _ = controllerSI.run(u0=uinit, t0=t0, Tend=Tend)
    uendFI, _ = controllerFI.run(u0=uinit, t0=t0, Tend=Tend)

    assert np.allclose(uendSI, uendFI), "Values at end time does not match!"

    errSI, errFI = abs(uendSI - P.u_exact(Tend)), abs(uendFI - P.u_exact(Tend))
    assert np.allclose(errSI, errFI), "Errors does not match!"


@pytest.mark.base
@pytest.mark.parametrize('case', [0, 1])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('QI', ['IE', 'LU'])
def testOrderAccuracy(case, M, QI):
    r"""
    In this test, the order of accuracy of the ``SemiImplicitDAE`` sweeper is tested for an index-1 DAE
    and an index-2 DAE of semi-explicit form.
    """

    from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.misc.HookClass_DAE import (
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    )
    from pySDC.helpers.stats_helper import get_sorted

    problem = {
        0: DiscontinuousTestDAE,
        1: simple_dae_1,
    }

    interval = {
        'DiscontinuousTestDAE': (1.0, 1.5),
        'simple_dae_1': (0.0, 0.4),
    }

    refOrderDiff = {
        'DiscontinuousTestDAE': 2 * M - 1,
        'simple_dae_1': 2 * M - 1,
    }

    # note that for index-2 DAEs there is order reduction in alg. variable
    refOrderAlg = {
        'DiscontinuousTestDAE': 2 * M - 1,
        'simple_dae_1': M,
    }

    hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
    description, controller_params = getTestSetup(problem[case], SemiImplicitDAE, hook_class)

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': QI,
    }
    description.update({'sweeper_params': sweeper_params})

    level_params = description['level_params']

    intervalCase = interval[problem[case].__name__]
    t0, Tend = intervalCase[0], intervalCase[-1]
    dt_list = np.logspace(-1.7, -1.0, num=5)

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
        orderDiff, refOrderDiff[problem[case].__name__], atol=1e0
    ), f"Expected order {refOrderDiff[problem[case].__name__]} in differential variable, got {orderDiff}"
    assert np.isclose(
        orderAlg, refOrderAlg[problem[case].__name__], atol=1e0
    ), f"Expected order {refOrderAlg[problem[case].__name__]} in algebraic variable, got {orderAlg}"
