import pytest
import os
import sys
import subprocess


@pytest.mark.base
def test_float_conversion():
    '''
    Method to test the float conversion by converting to bytes and back and by flipping bits where we know what the
    impact is. We try with 1000 random numbers, so we don't know how many times we get nan beforehand.
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


@pytest.mark.base
def test_fault_injection():
    from pySDC.projects.Resilience.fault_injection import FaultInjector

    # setup arguments for fault generation for van der Pol problem
    rnd_args = {'iteration': 3}
    args = {'time': 1.0, 'target': 0}
    injector = FaultInjector()
    injector.rnd_params = {
        'level_number': 1,
        'node': 3,
        'iteration': 3,
        'problem_pos': (2,),
        'bit': 64,
    }

    reference = {
        0: {
            'time': 1.0,
            'timestep': None,
            'level_number': 0,
            'iteration': 3,
            'node': 0,
            'problem_pos': [1],
            'bit': 48,
            'target': 0,
            'when': 'after',
        },
        1: {
            'time': 1.0,
            'timestep': None,
            'level_number': 0,
            'iteration': 3,
            'node': 3,
            'problem_pos': [0],
            'bit': 26,
            'target': 0,
            'when': 'after',
        },
        2: {
            'time': 1.0,
            'timestep': None,
            'level_number': 0,
            'iteration': 1,
            'node': 0,
            'problem_pos': [0],
            'bit': 0,
            'target': 0,
            'when': 'after',
        },
        3: {
            'time': 1.0,
            'timestep': None,
            'level_number': 0,
            'iteration': 1,
            'node': 1,
            'problem_pos': [0],
            'bit': 0,
            'target': 0,
            'when': 'after',
        },
    }

    # inject the faults
    for i in range(4):
        injector.add_fault(args=args, rnd_args=rnd_args)
        if i >= 1:  # switch to combination based adding
            injector.random_generator = i - 1
        assert (
            injector.faults[i].__dict__ == reference[i]
        ), f'Expected fault with parameters {reference[i]}, got {injector.faults[i].__dict__}!'


@pytest.mark.mpi4py
def test_fault_stats():
    """
    Test generation of fault statistics and their recovery rates
    """
    import numpy as np
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        AdaptivityStrategy,
        IterateStrategy,
        HotRodStrategy,
        run_vdp,
    )

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {4} python {__file__} --test-fault-stats".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        4,
    )

    vdp_stats = generate_stats(True)

    # test number of possible combinations for faults
    assert (
        vdp_stats.get_max_combinations() == 1536
    ), f"Expected 1536 possible combinations for faults in van der Pol problem, but got {vdp_stats.get_max_combinations()}!"

    recovered_reference = {
        'base': 1,
        'adaptivity': 2,
        'iterate': 1,
        'Hot Rod': 2,
    }
    vdp_stats.get_recovered()

    for strategy in vdp_stats.strategies:
        dat = vdp_stats.load(strategy, True)
        fixable_mask = vdp_stats.get_fixable_faults_only(strategy)
        recovered_mask = vdp_stats.get_mask(strategy=strategy, key='recovered', op='eq', val=True)

        assert all(fixable_mask[:-1] == [False, True, False]), "Error in generating mask of fixable faults"

        recovered = len(dat['recovered'][recovered_mask])
        crashed = len(dat['error'][dat['error'] == np.inf])  # on some systems the last run crashes...
        assert (
            recovered >= recovered_reference[strategy.name] - crashed
        ), f'Expected {recovered_reference[strategy.name]} recovered faults, but got {recovered} recovered faults in {strategy.name} strategy!'


def generate_stats(load=False):
    """
    Generate stats to check the recovery rate

    Args:
        load: Load the stats or generate them from scratch

    Returns:
        Object containing the stats
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

    np.seterr(all='warn')  # get consistent behaviour across platforms

    vdp_stats = FaultStats(
        prob=run_vdp,
        faults=[False, True],
        reload=load,
        recovery_thresh=1.1,
        num_procs=1,
        mode='random',
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        stats_path='data',
    )
    vdp_stats.run_stats_generation(runs=4, step=2)
    return vdp_stats


if __name__ == "__main__":
    if '--test-fault-stats' in sys.argv:
        generate_stats()
    else:
        test_fault_stats()
        test_fault_injection()
        test_float_conversion()
