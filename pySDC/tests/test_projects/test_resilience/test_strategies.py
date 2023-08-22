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
]
STRATEGY_NAMES_NONMPIONLY = ['adaptiveHR', 'HotRod']
LOGGER_LEVEL = 30


def single_test_vdp(strategy_name, useMPI, num_procs):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.Resilience.vdp import run_vdp
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
        'adaptivity': strategies.AdaptivityStrategy(useMPI=useMPI),
        'DIRK': strategies.DIRKStrategy(useMPI=useMPI),
        'adaptiveHR': strategies.AdaptiveHotRodStrategy(useMPI=useMPI),
        'iterate': strategies.IterateStrategy(useMPI=useMPI),
        'HotRod': strategies.HotRodStrategy(useMPI=useMPI),
        'explicitRK': strategies.ERKStrategy(useMPI=useMPI),
        'doubleAdaptivity': strategies.DoubleAdaptivityStrategy(useMPI=useMPI),
        'collocationRefinement': strategies.AdaptivityCollocationRefinementStrategy(useMPI=useMPI),
        'collocationDerefinement': strategies.AdaptivityCollocationDerefinementStrategy(useMPI=useMPI),
        'collocationType': strategies.AdaptivityCollocationTypeStrategy(useMPI=useMPI),
        'adaptivityAvoidRestarts': strategies.AdaptivityAvoidRestartsStrategy(useMPI=useMPI),
        'adaptivityInterpolation': strategies.AdaptivityInterpolationStrategy(useMPI=useMPI),
        'adaptivityQExtrapolation': strategies.AdaptivityExtrapolationWithinQStrategy(useMPI=useMPI),
        'base': strategies.BaseStrategy(useMPI=useMPI),
        'ESDIRK': strategies.ESDIRKStrategy(useMPI=useMPI),
    }

    if strategy_name in avail_strategies.keys():
        strategy = avail_strategies[strategy_name]
    else:
        raise NotImplementedError(f'Strategy \"{strategy_name}\" not implemented for this test!')

    prob = run_vdp
    controller_params = {'logger_level': LOGGER_LEVEL}
    stats, _, Tend = prob(
        custom_description=strategy.get_custom_description(problem=prob, num_procs=num_procs),
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
        ), f'Error in \"{strategy.name}\" strategy ({strategy_name})! Expected {key}={ref} but got {act}!'

    if comm:
        comm.Free()


@pytest.mark.mpi4py
@pytest.mark.parametrize('strategy_name', STRATEGY_NAMES)
def test_strategy_with_vdp_MPI(strategy_name, num_procs=1):
    single_test_vdp(strategy_name=strategy_name, useMPI=True, num_procs=num_procs)


@pytest.mark.base
@pytest.mark.parametrize('strategy_name', STRATEGY_NAMES + STRATEGY_NAMES_NONMPIONLY)
def test_strategy_with_vdp_nonMPI(strategy_name, num_procs=1):
    single_test_vdp(strategy_name=strategy_name, useMPI=False, num_procs=num_procs)


if __name__ == '__main__':
    for name in STRATEGY_NAMES + STRATEGY_NAMES_NONMPIONLY:
        test_strategy_with_vdp_nonMPI(name)
    for name in STRATEGY_NAMES:
        test_strategy_with_vdp_MPI(name)
