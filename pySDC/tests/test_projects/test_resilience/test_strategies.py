import pytest

STRATEGY_NAMES = [
    'doubleAdaptivity',
    'collocationType',
    'collocationRefinement',
    'collocationDerefinement',
    # 'adaptivityInterpolation',
    'adaptivityQExtrapolation',
    'adaptivityAvoidRestarts',
    'adaptivity',
    'iterate',
    'base',
    'DIRK',
    'explicitRK',
    'ESDIRK',
    'AdaptivityPolynomialError',
    'kAdaptivity',
]
STRATEGY_NAMES_NONMPIONLY = ['adaptiveHR', 'HotRod']
STRATEGY_NAMES_MPIONLY = ['ARK']
LOGGER_LEVEL = 30


def single_test(strategy_name, useMPI, num_procs):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    import pySDC.projects.Resilience.strategies as strategies
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD.Split(True)
    else:
        comm = None

    # load the strategy
    avail_strategies = {
        'adaptivity': strategies.AdaptivityStrategy,
        'DIRK': strategies.DIRKStrategy,
        'adaptiveHR': strategies.AdaptiveHotRodStrategy,
        'iterate': strategies.IterateStrategy,
        'HotRod': strategies.HotRodStrategy,
        'explicitRK': strategies.ERKStrategy,
        'doubleAdaptivity': strategies.DoubleAdaptivityStrategy,
        'collocationRefinement': strategies.AdaptivityCollocationRefinementStrategy,
        'collocationDerefinement': strategies.AdaptivityCollocationDerefinementStrategy,
        'collocationType': strategies.AdaptivityCollocationTypeStrategy,
        'adaptivityAvoidRestarts': strategies.AdaptivityAvoidRestartsStrategy,
        'adaptivityInterpolation': strategies.AdaptivityInterpolationStrategy,
        'adaptivityQExtrapolation': strategies.AdaptivityExtrapolationWithinQStrategy,
        'base': strategies.BaseStrategy,
        'ESDIRK': strategies.ESDIRKStrategy,
        'ARK': strategies.ARKStrategy,
        'AdaptivityPolynomialError': strategies.AdaptivityPolynomialError,
        'kAdaptivity': strategies.kAdaptivityStrategy,
    }

    if strategy_name in avail_strategies.keys():
        strategy = avail_strategies[strategy_name](useMPI=useMPI)
    else:
        raise NotImplementedError(f'Strategy \"{strategy_name}\" not implemented for this test!')

    if strategy_name in ['ARK']:
        from pySDC.projects.Resilience.Schroedinger import run_Schroedinger as prob
    else:
        from pySDC.projects.Resilience.vdp import run_vdp as prob
    controller_params = {'logger_level': LOGGER_LEVEL}
    custom_description = strategy.get_custom_description(problem=prob, num_procs=num_procs)

    if strategy_name == 'AdaptivityPolynomialError':
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

        custom_description['convergence_controllers'][AdaptivityPolynomialError]['e_tol'] = 1e-4

    stats, _, Tend = prob(
        custom_description=custom_description,
        hook_class=[LogGlobalErrorPostRun, LogWork],
        use_MPI=useMPI,
        custom_controller_params=controller_params,
        comm=comm,
    )

    # things we want to test
    tests = {
        'e': ('e_global_post_run', max),
        'k_newton': ('work_newton', sum),
    }

    for key, val in tests.items():
        act = val[1]([me[1] for me in get_sorted(stats, type=val[0], comm=comm)])
        ref = strategy.get_reference_value(prob, val[0], val[1], num_procs)

        assert np.isclose(
            ref, act, rtol=1e-2
        ), f'Error in \"{strategy.name}\" strategy ({strategy_name})! Got {key}={act} but expected {ref}!'

    if comm:
        comm.Free()


@pytest.mark.mpi4py
@pytest.mark.parametrize('strategy_name', STRATEGY_NAMES + STRATEGY_NAMES_MPIONLY)
def test_strategy_MPI(strategy_name, num_procs=1):
    single_test(strategy_name=strategy_name, useMPI=True, num_procs=num_procs)


@pytest.mark.base
@pytest.mark.parametrize('strategy_name', STRATEGY_NAMES + STRATEGY_NAMES_NONMPIONLY)
def test_strategy_nonMPI(strategy_name, num_procs=1):
    single_test(strategy_name=strategy_name, useMPI=False, num_procs=num_procs)


if __name__ == '__main__':
    for name in STRATEGY_NAMES + STRATEGY_NAMES_NONMPIONLY:
        test_strategy_nonMPI(name)
    for name in STRATEGY_NAMES:
        test_strategy_MPI(name)
