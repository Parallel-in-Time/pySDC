import pytest

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}

HEAT_PARAMS = {
    'nvars': 63,
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}

AC_PARAMS = {
    'nvars': (32, 32),
    'eps': 0.04,
    'newton_maxiter': 100,
    'newton_tol': 1e-12,
    'lin_tol': 1e-12,
    'lin_maxiter': 200,
    'radius': 0.25,
}


def sweeper_params(**extra):
    params = dict(SWEEPER_PARAMS)
    params.update(extra)
    return params


def run(problem_class, problem_params, sweeper_class, sweeper_params, dt, nsteps, maxiter, num_procs=1, restol=-1):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, stats, prob


@pytest.mark.base
@pytest.mark.parametrize('QI', ['IE', 'LU', 'MIN-SR-S'])
@pytest.mark.parametrize('maxiter', [1, 3, 8])
def test_delta_form_matches_generic_implicit(QI, maxiter):
    """The delta form is an algebraic rewrite, so it must reproduce the standard sweep exactly."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, sweeper_params(QI=QI), 1e-2, 2, maxiter)
    u_std, _, _ = run(args[0], args[1], generic_implicit, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-13, f'delta form deviates by {abs(u_std - u_delta):.3e}'


@pytest.mark.base
def test_zero_diagonal_preconditioner():
    """A Picard preconditioner has a zero diagonal, exercising the explicit branch."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, sweeper_params(QI='PIC'), 1e-3, 1, 3)
    u_std, _, _ = run(args[0], args[1], generic_implicit, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-13


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_correction_precision_storage(precision):
    """Storing the small quantities at reduced precision must not change the answer much."""
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u64, _, prob = run(heatNd_unforced, HEAT_PARAMS, delta_implicit, sweeper_params(), 1e-2, 2, 6)
    u_red, _, _ = run(
        heatNd_unforced,
        HEAT_PARAMS,
        delta_implicit,
        sweeper_params(correction_precision=np.dtype(precision)),
        1e-2,
        2,
        6,
    )
    tolerance = 1e-10 if precision == 'float32' else 1e-3
    assert abs(u64 - u_red) < tolerance


@pytest.mark.base
def test_linear_implicit_reuses_stock_solve_system():
    """For a linear operator the stock solve_system already solves the correction equation."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, 1e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(linear_implicit=True), *args[2:])
    assert abs(u_std - u_delta) < 1e-12


@pytest.mark.base
def test_linear_implicit_handles_affine_operator():
    """An affine operator needs the f(0, t) shift, which the sweeper subtracts."""
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit
    from pySDC.projects.DeltaSDC.tests.affine_problem import heat_affine

    args = (heat_affine, HEAT_PARAMS, 1e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(linear_implicit=True), *args[2:])
    assert abs(u_std - u_delta) < 1e-12


@pytest.mark.base
def test_mlsdc_tau_correction():
    """The residual must pick up the FAS tau term on coarse levels."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    results = {}
    for sweeper in [generic_implicit, delta_implicit]:
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': dict(HEAT_PARAMS, nvars=[63, 31]),
            'sweeper_class': sweeper,
            'sweeper_params': sweeper_params(),
            'level_params': {'restol': -1, 'dt': 1e-2},
            'step_params': {'maxiter': 5},
            'space_transfer_class': mesh_to_mesh,
            'space_transfer_params': {'iorder': 2, 'rorder': 2},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        results[sweeper], _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2e-2)

    assert abs(results[generic_implicit] - results[delta_implicit]) < 1e-12


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [2, 4])
def test_parallel_in_time(num_procs):
    """Nothing in the controller changes, so block-parallel runs must agree with the serial one."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u_ref, _, _ = run(heatNd_unforced, HEAT_PARAMS, generic_implicit, sweeper_params(), 1e-2, 4, 20, restol=1e-11)
    u_par, _, _ = run(
        heatNd_unforced,
        HEAT_PARAMS,
        delta_implicit,
        sweeper_params(),
        1e-2,
        4,
        20,
        num_procs=num_procs,
        restol=1e-11,
    )
    assert abs(u_ref - u_par) < 1e-9


@pytest.mark.base
@pytest.mark.parametrize('maxiter', [1, 4, 10])
def test_delta_imex_matches_imex_1st_order(maxiter):
    """The IMEX splitting survives because the correction equation needs no Jacobian."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.DeltaSDC.sweepers import delta_imex_1st_order

    params = sweeper_params(QI='IE', QE='EE')
    args = (heatNd_forced, HEAT_PARAMS, params, 5e-2, 2, maxiter)
    u_std, _, _ = run(args[0], args[1], imex_1st_order, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_imex_1st_order, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-12


@pytest.mark.base
def test_delta_imex_correction_precision_and_linear_implicit():
    """Exercise the reduced-precision storage and the linear-implicit path for IMEX."""
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.DeltaSDC.sweepers import delta_imex_1st_order

    args = (heatNd_forced, HEAT_PARAMS, 5e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], imex_1st_order, sweeper_params(QI='IE', QE='EE'), *args[2:])
    u_red, _, _ = run(
        args[0],
        args[1],
        delta_imex_1st_order,
        sweeper_params(QI='IE', QE='EE', correction_precision=np.dtype('float32'), linear_implicit=True),
        *args[2:],
    )
    assert abs(u_std - u_red) < 1e-8


@pytest.mark.base
def test_solve_system_delta_path_is_preferred():
    """When the problem offers a correction solve, the sweeper must use it."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems import allencahn_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    calls = {'n': 0}
    original = allencahn_delta.solve_system_delta

    def counting(self, *args, **kwargs):
        calls['n'] += 1
        return original(self, *args, **kwargs)

    allencahn_delta.solve_system_delta = counting
    try:
        u_std, _, _ = run(allencahn_fullyimplicit, AC_PARAMS, generic_implicit, sweeper_params(), 4e-4, 2, 8)
        params = dict(AC_PARAMS, krylov_tol=1e-12, newton_rtol=1e-12)
        u_delta, _, _ = run(allencahn_delta, params, delta_implicit, sweeper_params(), 4e-4, 2, 8)
    finally:
        allencahn_delta.solve_system_delta = original

    assert calls['n'] > 0, 'solve_system_delta was never called'
    assert abs(u_std - u_delta) < 1e-10


@pytest.mark.base
def test_nonlinear_fallback_without_correction_solve():
    """Without a correction solve the sweeper falls back to the substitution, still exactly."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    args = (allencahn_fullyimplicit, AC_PARAMS, 4e-4, 2, 8)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(), *args[2:])
    assert abs(u_std - u_delta) < 1e-11
