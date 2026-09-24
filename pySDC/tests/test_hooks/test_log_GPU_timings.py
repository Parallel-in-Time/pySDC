import math

import pytest


def run_with_GPU_timings(num_steps=2, dt=0.1):
    """Run a small problem on the GPU with `GPUTimings` added to the controller's default hooks."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_GPU_timings import GPUTimings
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    description = {
        'problem_class': polynomial_testequation,
        'problem_params': {'degree': 4, 'useGPU': True},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3},
        'level_params': {'dt': dt, 'restol': -1},
        'step_params': {'maxiter': 2},
    }
    # `hook_class` is appended to `[DefaultHooks, CPUTimings]` rather than replacing them, so the
    # CPU twin records alongside and the two sets of timings can be compared against each other.
    controller_params = {'logger_level': 30, 'hook_class': [GPUTimings], 'mssdc_jac': False}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=num_steps * dt)
    return stats


@pytest.mark.cupy
def test_GPU_timings_record_what_the_CPU_ones_do():
    """`GPUTimings` only swaps how an interval is measured, so it has to record the same events."""
    from pySDC.helpers.stats_helper import get_sorted

    stats = run_with_GPU_timings()

    recorded = {key.type for key in stats}
    on_cpu = {entry for entry in recorded if entry.startswith('timing_')}
    on_gpu = {entry for entry in recorded if entry.startswith('GPU_timing_')}

    assert on_cpu, 'the default CPU timings hook recorded nothing, so there is nothing to compare against'
    assert {f'GPU_{entry}' for entry in on_cpu} == on_gpu, (
        f'the GPU hook did not record the same events as the CPU one: only on the CPU '
        f'{sorted(entry for entry in on_cpu if f"GPU_{entry}" not in on_gpu)}, only on the GPU '
        f'{sorted(entry for entry in on_gpu if entry.removeprefix("GPU_") not in on_cpu)}'
    )

    for entry in on_gpu:
        for time, value in get_sorted(stats, type=entry):
            assert math.isfinite(value) and value >= 0, f'{entry} at {time=} is not a duration: {value}'

    # Everything else can legitimately round to zero -- CUDA events resolve to about half a
    # microsecond, and a sweep of a degree-4 polynomial is not much work -- but the whole run
    # takes measurable time, so a zero here means the events were never really read.
    duration = get_sorted(stats, type='GPU_timing_run')
    assert len(duration) == 1, f'expected exactly one run timing, got {len(duration)}'
    assert duration[0][1] > 0, 'the run was measured as taking no time at all'
