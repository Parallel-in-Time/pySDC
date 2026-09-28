import pytest

# `test_strategies` only runs the strategies on Lorenz. These are the other problems of the project,
# each with the strategies whose per-problem settings it has, run for a few steps. RBC comes first, so
# that everything after it would notice if running it changed library classes again.
TENDS = {'run_RBC': 2e-3, 'run_AC': 2e-4, 'run_GS': 2.0, 'run_quench': 20.0, 'run_Schroedinger': 0.1}
# RBC starts at t=0, where it needs no reference solution, on a grid small enough to be quick
PROBLEM_ARGS = {'run_RBC': {'t0': 0}}
PROBLEM_PARAMS = {'run_RBC': {'nx': 32, 'nz': 16}}
SDC_STRATEGIES = [
    'BaseStrategy',
    'AdaptivityStrategy',
    'IterateStrategy',
    'kAdaptivityStrategy',
    'HotRodStrategy',
    'DoubleAdaptivityStrategy',
    'AdaptivityAvoidRestartsStrategy',
]
COMBINATIONS = (
    [(problem, strategy) for problem in ['run_RBC', 'run_AC', 'run_GS', 'run_quench'] for strategy in SDC_STRATEGIES]
    + [(problem, 'AdaptivityPolynomialError') for problem in ['run_AC', 'run_GS', 'run_quench']]
    + [(problem, 'AdaptivityExtrapolationWithinQStrategy') for problem in ['run_AC', 'run_quench']]
    + [(problem, 'AdaptivityCollocationTypeStrategy') for problem in ['run_AC', 'run_quench']]
    + [(problem, 'ARKStrategy') for problem in ['run_RBC', 'run_AC', 'run_GS', 'run_Schroedinger']]
    + [('run_quench', strategy) for strategy in ['DIRKStrategy', 'ESDIRKStrategy']]
    + [('run_Schroedinger', strategy) for strategy in ['BaseStrategy', 'AdaptivityStrategy']]
)


def get_problem(name):
    from pySDC.projects.Resilience.AC import run_AC
    from pySDC.projects.Resilience.GS import run_GS
    from pySDC.projects.Resilience.RBC import run_RBC
    from pySDC.projects.Resilience.quench import run_quench
    from pySDC.projects.Resilience.Schroedinger import run_Schroedinger

    return {me.__name__: me for me in [run_AC, run_GS, run_RBC, run_quench, run_Schroedinger]}[name]


def get_description(problem, strategy):
    from pySDC.projects.Resilience.strategies import merge_descriptions

    return merge_descriptions(
        strategy.get_custom_description(problem, num_procs=1),
        {'problem_params': PROBLEM_PARAMS.get(problem.__name__, {})},
    )


@pytest.mark.mpi4py
@pytest.mark.parametrize('problem_name, strategy_name', COMBINATIONS)
def test_problem_with_strategy(problem_name, strategy_name):
    import numpy as np
    import pySDC.projects.Resilience.strategies as strategies

    problem = get_problem(problem_name)
    strategy = getattr(strategies, strategy_name)()

    stats, controller, crash = problem(
        custom_description=get_description(problem, strategy),
        Tend=TENDS[problem_name],
        custom_controller_params={'logger_level': 30},
        **PROBLEM_ARGS.get(problem_name, {}),
    )

    assert not crash, f'{strategy_name} crashed on {problem_name}'
    uend = controller.MS[0].levels[0].uend
    assert np.all(np.isfinite(uend)), f'{strategy_name} produced a non-finite solution on {problem_name}'


@pytest.mark.mpi4py
def test_RBC_leaves_library_classes_alone():
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorNonMPI,
    )
    from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
    from pySDC.projects.Resilience.strategies import HotRodStrategy
    from pySDC.projects.Resilience.RBC import run_RBC
    import pySDC.projects.Resilience.GS  # noqa: F401 (used to patch Gray-Scott on import)

    def get_methods():
        return (
            RayleighBenard.u_exact,
            grayscott_imex_diffusion.u_exact,
            imex_1st_order_efficient.compute_residual,
            EstimateExtrapolationErrorNonMPI.get_extrapolated_error,
        )

    before = get_methods()
    run_RBC(
        custom_description=get_description(run_RBC, HotRodStrategy()),
        Tend=TENDS['run_RBC'],
        custom_controller_params={'logger_level': 30},
        **PROBLEM_ARGS['run_RBC'],
    )
    assert get_methods() == before, 'Running RBC changed classes that other problems use'


@pytest.mark.base
def test_heat_converges():
    from pySDC.projects.Resilience.heat import run_heat
    from pySDC.projects.Resilience.strategies import BaseStrategy

    Tend = 0.1
    errors = []
    for dt in [0.05, 0.025]:
        description = BaseStrategy().get_custom_description(run_heat, num_procs=1)
        description['level_params']['dt'] = dt
        _, controller, _ = run_heat(custom_description=description, Tend=Tend)
        level = controller.MS[0].levels[0]
        errors.append(abs(level.uend - level.prob.u_exact(Tend)))

    assert errors[1] < errors[0] / 4, f'Halving the step size did not reduce the error enough: {errors}'
