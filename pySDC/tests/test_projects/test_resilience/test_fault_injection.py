import pytest


@pytest.mark.base
def test_float_conversion():
    '''
    Method to test the float conversion by converting to bytes and back and by flipping bits where we know what the
    impact is. We try with 1000 random numbers, so we don't know how many times we get nan before hand.
    '''
    import numpy as np
    from pySDC.projects.Resilience.fault_injection import FaultInjector

    # Try the conversion between floats and bytes
    injector = FaultInjector()
    exp = [-1, 2, 256]
    bit = [0, 11, 8]
    nan_counter = 0
    num_tests = int(1e3)
    for i in range(num_tests):
        # generate a random number almost between the full range of python float
        rand = np.random.uniform(low=-1.797693134862315e307, high=1.797693134862315e307, size=1)[0]
        # convert to bytes and back
        res = injector.to_float(injector.to_binary(rand))
        assert np.isclose(res, rand), f"Conversion between bytes and float failed for {rand}: result: {res}"

        # flip some exponent bits
        for i in range(len(exp)):
            res = injector.flip_bit(rand, bit[i]) / rand
            if np.isfinite(res):
                assert exp[i] in [
                    res,
                    1.0 / res,
                ], f'Bitflip failed: expected ratio: {exp[i]}, got: {res:.2e} or \
{1./res:.2e}'
            else:
                nan_counter += 1
    if nan_counter > 0:
        print(f'When flipping bits, we got nan {nan_counter} times out of {num_tests} tests')


@pytest.mark.slow
@pytest.mark.base
def test_fault_stats():
    """
    Test generation of fault statistics and their recovery rates
    """
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        AdaptivityStrategy,
        IterateStrategy,
        HotRodStrategy,
        run_vdp,
    )
    import matplotlib.pyplot as plt
    import numpy as np

    vdp_stats = FaultStats(
        prob=run_vdp,
        faults=[False, True],
        reload=False,
        recovery_thresh=1.1,
        num_procs=1,
        mode='random',
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        stats_path='data',
    )

    # test number of possible combinations for faults
    assert (
        vdp_stats.get_max_combinations() == 1536
    ), f"Expected 1536 possible combinations for faults in van der Pol problem, but got {vdp_stats.get_max_combinations()}!"

    # test recovery rate
    vdp_stats.run_stats_generation(runs=4, step=4)
    recovered_reference = {
        'base': 1,
        'adaptivity': 2,
        'iterate': 1,
        'Hot Rod': 2,
    }
    vdp_stats.get_recovered()

    for strategy in vdp_stats.strategies:
        dat = vdp_stats.load(strategy, True)
        recovered = len(dat['recovered'][dat['recovered'] == True])
        assert (
            recovered == recovered_reference[strategy.name]
        ), f'Expected {recovered_reference[strategy.name]} recovered faults, but got {recovered} recovered faults in {strategy.name} strategy!'


if __name__ == "__main__":
    test_fault_stats()
    test_float_conversion()
