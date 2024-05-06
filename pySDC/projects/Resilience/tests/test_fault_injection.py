import pytest
import os
import sys
import subprocess
import numpy as np


def get_random_float():
    """
    Get a random float64 number in the full range.

    Returns:
        float: Random float
    """
    rand = 0.0
    while np.isclose(rand, 0.0, atol=1e-12):
        rand = np.random.uniform(low=np.finfo(float).min / 1e1, high=np.finfo(float).max / 1e1, size=1)[0]
    return rand


@pytest.mark.base
def test_float_conversion():
    """
    Method to test the float conversion by converting to bytes and back and by flipping bits where we know what the
    impact is. We try with 1000 random numbers, so we don't know how many times we get nan beforehand.
    """
    from pySDC.projects.Resilience.fault_injection import FaultInjector

    # Try the conversion between floats and bytes
    injector = FaultInjector()
    exp = [-1, 2, 256]
    bit = [0, 11, 8]
    nan_counter = 0
    num_tests = int(1e3)
    for i in range(num_tests):
        # generate a random number almost between the full range of python float
        rand = get_random_float()
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
                ], f'Bitflip failed: expected ratio: {exp[i]}, got: {res:.2e} or {1./res:.2e}'
            else:
                nan_counter += 1
    if nan_counter > 0:
        print(f'When flipping bits, we got nan {nan_counter} times out of {num_tests} tests')


@pytest.mark.base
def test_complex_conversion():
    """
    Test conversion of complex numbers to and from binary
    """
    from pySDC.projects.Resilience.fault_injection import FaultInjector

    injector = FaultInjector()
    num_tests = int(1e3)
    for _i in range(num_tests):
        rand_complex = get_random_float() + get_random_float() * 1j

        # convert to bytes and back
        res = injector.to_float(injector.to_binary(rand_complex))
        assert np.isclose(
            res, rand_complex
        ), f"Conversion between bytes and float failed for {rand_complex}: result: {res}"


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
        'rank': 4,
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
            'rank': 3,
        },
        1: {
            'time': 1.0,
            'timestep': None,
            'level_number': 0,
            'iteration': 3,
            'node': 2,
            'problem_pos': [0],
            'bit': 7,
            'target': 0,
            'when': 'after',
            'rank': 1,
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
            'rank': 0,
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
            'rank': 0,
        },
    }

    # inject the faults
    for i in range(4):
        injector.add_fault(args=args, rnd_args=rnd_args)
        if i >= 1:  # switch to combination based adding
            injector.random_generator = i - 1
        for key in reference[i].keys():
            assert (
                injector.faults[i].__dict__[key] == reference[i][key]
            ), f'Expected fault with parameter {key}={reference[i][key]}, got {injector.faults[i].__dict__[key]} in fault number {i}!'


@pytest.mark.mpi4py
@pytest.mark.slow
@pytest.mark.parametrize("numprocs", [5])
def test_fault_stats(numprocs):
    """
    Test generation of fault statistics and their recovery rates
    """
    import numpy as np

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {numprocs} python {__file__} --test-fault-stats".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        numprocs,
    )

    stats = generate_stats(True)

    # test number of possible combinations for faults
    expected_max_combinations = 3840
    assert (
        stats.get_max_combinations() == expected_max_combinations
    ), f"Expected {expected_max_combinations} possible combinations for faults in van der Pol problem, but got {stats.get_max_combinations()}!"

    recovered_reference = {
        'base': 1,
        'adaptivity': 2,
        'iterate': 1,
        'Hot Rod': 2,
        'adaptivity_coll': 0,
        'double_adaptivity': 0,
    }
    stats.get_recovered()

    for strategy in stats.strategies:
        dat = stats.load(strategy=strategy, faults=True)
        fixable_mask = stats.get_fixable_faults_only(strategy)
        recovered_mask = stats.get_mask(strategy=strategy, key='recovered', op='eq', val=True)
        index = stats.get_index(mask=fixable_mask)

        assert all(fixable_mask[:-1] == [False, True, False]), "Error in generating mask of fixable faults"
        assert all(index == [1, 3]), "Error when converting to  index"

        combinations = np.array(stats.get_combination_counts(dat, keys=['bit'], mask=fixable_mask))
        assert all(combinations == [1.0, 1.0]), "Error when counting combinations"

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
    from pySDC.projects.Resilience.strategies import (
        BaseStrategy,
        AdaptivityStrategy,
        IterateStrategy,
        HotRodStrategy,
    )
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
    )
    from pySDC.projects.Resilience.Lorenz import run_Lorenz

    np.seterr(all='warn')  # get consistent behaviour across platforms

    stats = FaultStats(
        prob=run_Lorenz,
        faults=[False, True],
        reload=load,
        recovery_thresh=1.1,
        num_procs=1,
        mode='random',
        strategies=[
            BaseStrategy(),
            AdaptivityStrategy(),
            IterateStrategy(),
            HotRodStrategy(),
        ],
        stats_path='data',
    )
    stats.run_stats_generation(runs=4, step=2)
    return stats


if __name__ == "__main__":
    if '--test-fault-stats' in sys.argv:
        generate_stats()
    else:
        test_complex_conversion()
        test_fault_stats(5)
        test_fault_injection()
        test_float_conversion()
