import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('strategy_name', ['AdaptivityStrategy', 'BaseStrategy'])
def test_record_work_precision(strategy_name, tmp_path, monkeypatch):
    """Tightening the precision parameter has to buy accuracy with work, and the data has to survive a reload."""
    import numpy as np
    from mpi4py import MPI
    import pySDC.projects.Resilience.strategies as strategies
    from pySDC.projects.Resilience.vdp import run_vdp
    from pySDC.projects.Resilience.work_precision import record_work_precision, load, extract_data

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data' / 'work_precision').mkdir(parents=True)

    strategy = getattr(strategies, strategy_name)(useMPI=True)
    param_range = {'AdaptivityStrategy': [1e-4, 1e-7], 'BaseStrategy': [4e-2, 1e-2]}[strategy_name]
    args = {'problem': run_vdp, 'strategy': strategy, 'num_procs': 1, 'handle': 'test'}

    record_work_precision(**args, comm_world=MPI.COMM_WORLD, param_range=param_range, Tend=2.0)

    work, precision = extract_data(load(**args), work_key='k_SDC', precision_key='e_global')
    assert work[1] > work[0], f'Expected more iterations for the tighter parameter, got {work}'
    assert precision[1] < precision[0], f'Expected a smaller error for the tighter parameter, got {precision}'


@pytest.mark.mpi4py
@pytest.mark.parametrize(
    'mode',
    [
        'regular',
        'step_size_limiting',
        'dynamic_restarts',
        'compare_strategies',
        'RK_comp',
        'parallel_efficiency',
        'parallel_efficiency_dt',
        'parallel_efficiency_dt_k',
        'interpolate_between_restarts',
        'diagonal_SDC',
        'vdp_stiffness-10',
        'inexactness',
        'compare_adaptivity',
        'preconditioners',
        'RK_comp_high_order',
        'avoid_restarts',
    ],
)
@pytest.mark.parametrize('problem_name', ['run_vdp', 'run_quench', 'run_Schroedinger'])
def test_get_configs(mode, problem_name):
    """The configurations name strategies and imports that have to keep existing for the paper plots to be rerun."""
    from pySDC.projects.Resilience.strategies import Strategy
    import pySDC.projects.Resilience.work_precision as work_precision

    configs = work_precision.get_configs(mode, getattr(work_precision, problem_name))

    assert len(configs) > 0, f'No configurations for mode {mode!r}'
    for config in configs.values():
        assert all(isinstance(me, Strategy) for me in config['strategies'])
