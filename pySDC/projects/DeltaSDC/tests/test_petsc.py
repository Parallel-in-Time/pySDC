import pytest

try:
    import petsc4py  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f'PETSc unavailable: {exc}', allow_module_level=True)

PETSC_PARAMS = {
    'nvars': 127,
    'lambda0': 2.0,
    'nu': 1,
    'interval': (-5.0, 5.0),
    'lsol_tol': 1e-12,
    'nlsol_tol': 1e-12,
    'lsol_maxiter': 200,
    'nlsol_maxiter': 50,
}

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}


def _run(problem_class, problem_params, sweeper_class, maxiter, dt=0.1, nsteps=2):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, prob


@pytest.mark.petsc
def test_binomial_increment_is_cancellation_free():
    """Every term must carry a factor delta, so the increment vanishes with the correction."""
    from pySDC.projects.DeltaSDC.problems_petsc import binomial_increment

    base = 0.7
    previous = None
    for delta in [1e-2, 1e-4, 1e-6]:
        value = binomial_increment(base, delta, 2)
        assert value == pytest.approx((base + delta) ** 2 - base**2, rel=1e-12)
        if previous is not None:
            assert abs(value) < abs(previous)
        previous = value


@pytest.mark.petsc
@pytest.mark.parametrize('maxiter', [3, 6, 10])
def test_correction_solve_matches_stock_path(maxiter):
    """The correction solve must reproduce the stock PETSc path."""
    from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import petsc_fisher_fullyimplicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems_petsc import petsc_fisher_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u_stock, _ = _run(petsc_fisher_fullyimplicit, PETSC_PARAMS, generic_implicit, maxiter)
    u_delta, _ = _run(petsc_fisher_delta, PETSC_PARAMS, delta_implicit, maxiter)
    assert abs(u_stock - u_delta) < 1e-12, f'deviates by {abs(u_stock - u_delta):.3e}'


@pytest.mark.petsc
def test_emulated_reduced_precision_retains_accuracy():
    """Capping the information at float32 must not move the answer.

    PETSc's scalar type is fixed at build time, so this is emulation: values are rounded through
    float32 while the arithmetic stays at the backend scalar type.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import petsc_fisher_fullyimplicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems_petsc import petsc_fisher_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u_stock, _ = _run(petsc_fisher_fullyimplicit, PETSC_PARAMS, generic_implicit, 10)
    u_emulated, prob = _run(petsc_fisher_delta, dict(PETSC_PARAMS, solve_precision=np.float32), delta_implicit, 10)

    assert prob.solve_precision == np.dtype('float32')
    assert abs(u_stock - u_emulated) < 1e-6, f'deviates by {abs(u_stock - u_emulated):.3e}'


@pytest.mark.petsc
def test_correction_shrinks_so_reduced_precision_is_safe():
    """The unknown handed to SNES must shrink with the sweeps, which is the whole premise."""
    import numpy as np
    from pySDC.projects.DeltaSDC.problems_petsc import petsc_fisher_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    magnitudes = []
    original = petsc_fisher_delta.solve_system_delta

    def spy(self, r, factor, base, f_base, t):
        delta = original(self, r, factor, base, f_base, t)
        magnitudes.append(abs(delta))
        return delta

    petsc_fisher_delta.solve_system_delta = spy
    try:
        _run(petsc_fisher_delta, PETSC_PARAMS, delta_implicit, 10, nsteps=1)
    finally:
        petsc_fisher_delta.solve_system_delta = original

    assert len(magnitudes) > 5
    assert min(magnitudes) < 1e-3 * max(
        magnitudes
    ), f'correction did not shrink: max {max(magnitudes):.2e}, min {min(magnitudes):.2e}'
